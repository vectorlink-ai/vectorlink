{mkShell, clang, rust-bin, openssl}:
mkShell {
  buildInputs = [
    clang
    openssl
    (rust-bin.nightly.latest.default.override {
      extensions = [ "rust-src" "rust-analyzer" ];
    })
  ];

  shellHook = ''
export RUSTFLAGS="-C target-feature=+avx2,+f16c,+fma"
'';
}
