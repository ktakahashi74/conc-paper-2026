mod paper_plots;
mod sim;

use std::env;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let _ = args; // consumed by paper_plots::main() via env::args()

    if let Err(err) = paper_plots::main() {
        eprintln!("paper plots failed: {err}");
        std::process::exit(1);
    }
}
