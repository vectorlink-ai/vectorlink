{pkgs, mkShell, clang, rust-bin, openssl, pkg-config}:
mkShell {
  buildInputs = [
    clang
    pkg-config
    openssl
    (rust-bin.nightly."2024-10-17".default.override {
      extensions = [ "rustfmt" "rust-src" "rust-analyzer" ];
      targets = [
        "aarch64-apple-darwin"
        "aarch64-unknown-linux-gnu"
        "x86_64-unknown-linux-gnu"
      ];
    })
  ];

  shellHook = if pkgs.system == "x86_64-linux" then ''
    export RUSTFLAGS="-C target-feature=+avx2,+f16c,+fma,+aes,+sse2"
  '' else if pkgs.system == "aarch64-linux" then ''
    export RUSTFLAGS="-C target-feature=+neon"
  '' else if pkgs.system == "aarch64-darwin" then ''
    export RUSTFLAGS="-C target-feature=+neon"
  '' else throw "Unknown system: ${pkgs.system}";
}
