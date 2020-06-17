use crate::NarrowStackedExpander;
use crate::{
    utils, Config, GPUContext, Layer, LayerOutput, NSEResult, ReplicaId, Sealer, TreeOptions, GPU,
};
use std::sync::mpsc;
use std::sync::{Condvar, Mutex, MutexGuard};
use std::thread;

type SealerInput = (ReplicaId, usize, Layer);

pub struct GpuPool {
    gpus: Vec<mpsc::Sender<(SealerInput, mpsc::Sender<LayerOutput>)>>,
}

impl GpuPool {
    pub fn new(config: Config, tree_options: TreeOptions) -> NSEResult<Self> {
        let mut gpus = Vec::new();

        let tree_enabled = if let TreeOptions::Enabled { rows_to_discard: _ } = tree_options {
            true
        } else {
            false
        };

        for dev in utils::all_devices()? {
            let (fn_tx, fn_rx): (
                mpsc::Sender<(SealerInput, mpsc::Sender<LayerOutput>)>,
                mpsc::Receiver<(SealerInput, mpsc::Sender<LayerOutput>)>,
            ) = mpsc::channel();
            gpus.push(fn_tx);
            thread::spawn(move || {
                let ctx = GPUContext::new(dev, config.clone(), tree_options.clone()).unwrap();
                let mut gpu = GPU::new(ctx, config.clone()).unwrap();
                loop {
                    let (inp, sender) = fn_rx.recv().unwrap();
                    let sealer =
                        Sealer::new(config.clone(), inp.0, inp.1, inp.2, &mut gpu, tree_enabled)
                            .unwrap();
                    for output in sealer {
                        sender.send(output.unwrap()).unwrap();
                    }
                }
            });
        }
        Ok(GpuPool { gpus })
    }

    pub fn run_on_gpu(&self, inp: SealerInput) -> mpsc::Receiver<LayerOutput> {
        let (tx, rx): (mpsc::Sender<LayerOutput>, mpsc::Receiver<LayerOutput>) = mpsc::channel();
        self.gpus[0].send((inp, tx)).unwrap();
        rx
    }
}
