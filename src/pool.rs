use crate::{Config, NSEResult, GPU};
use std::sync::{Condvar, Mutex, MutexGuard};

pub struct GpuPool {
    lock: Mutex<()>,
    cond: Condvar,
    gpus: Vec<Mutex<GPU>>,
}

impl GpuPool {
    pub fn new(config: Config) -> NSEResult<Self> {
        Ok(GpuPool {
            lock: Mutex::new(()),
            cond: Condvar::new(),
            gpus: Vec::new(),
        })
    }

    pub fn run_on_gpu<F>(&self, f: F)
    where
        F: FnOnce(MutexGuard<GPU>) -> (),
    {
        let mut lock = self.lock.lock().unwrap();
        loop {
            for gpu in self.gpus.iter() {
                match gpu.try_lock() {
                    Ok(guard) => {
                        drop(lock);
                        f(guard);
                        self.cond.notify_one();
                        return;
                    }
                    Err(_) => {}
                }
            }

            lock = self.cond.wait(lock).unwrap();
        }
    }
}
