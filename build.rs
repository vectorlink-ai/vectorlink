use std::process::exit;

macro_rules! check_feature {
    ($feature: literal) => {
        if !cfg!(target_feature = $feature) {
            println!("cargo:warning={} is required but not enabled.", $feature);
            exit(1);
        }
    };
}

fn main() {
    check_feature!("avx2");
    check_feature!("f16c");
    check_feature!("fma");
}
