use crate::NarrowStackedExpander;
use crate::{
    utils, Config, GPUContext, Layer, LayerOutput, NSEResult, ReplicaId, Sealer, TreeOptions, GPU,
};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

type SealerInput = (ReplicaId, usize, Layer);

struct GpuWorker {
    busy: Arc<Mutex<()>>,
    channel: mpsc::Sender<(SealerInput, mpsc::Sender<LayerOutput>)>,
}

pub struct GpuPool {
    lock: Mutex<()>,
    cond: Arc<Condvar>,
    workers: Vec<GpuWorker>,
}

impl GpuPool {
    pub fn new(config: Config, tree_options: TreeOptions) -> NSEResult<Self> {
        let mut workers = Vec::new();
        let cond = Arc::new(Condvar::new());

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
            let busy = Arc::new(Mutex::new(()));
            workers.push(GpuWorker {
                channel: fn_tx,
                busy: Arc::clone(&busy),
            });
            let cond = Arc::clone(&cond);
            thread::spawn(move || {
                let ctx = GPUContext::new(dev, config.clone(), tree_options.clone()).unwrap();
                let mut gpu = GPU::new(ctx, config.clone()).unwrap();
                loop {
                    let (inp, sender) = fn_rx.recv().unwrap();
                    let lock = busy.lock().unwrap(); // Flag worker as busy
                    let sealer =
                        Sealer::new(config.clone(), inp.0, inp.1, inp.2, &mut gpu, tree_enabled)
                            .unwrap();
                    for output in sealer {
                        sender.send(output.unwrap()).unwrap();
                    }
                    drop(lock);
                    cond.notify_one(); // Notify that one GPU is not busy anymore
                }
            });
        }
        Ok(GpuPool {
            workers,
            lock: Mutex::new(()),
            cond,
        })
    }

    pub fn run_on_gpu(&self, inp: SealerInput) -> mpsc::Receiver<LayerOutput> {
        let mut lock = self.lock.lock().unwrap();
        loop {
            for worker in self.workers.iter() {
                match worker.busy.try_lock() {
                    Ok(_) => {
                        let (tx, rx): (mpsc::Sender<LayerOutput>, mpsc::Receiver<LayerOutput>) =
                            mpsc::channel();
                        worker.channel.send((inp, tx)).unwrap();
                        return rx;
                    }
                    Err(_) => {}
                }
            }
            lock = self.cond.wait(lock).unwrap();
        }
    }
}
