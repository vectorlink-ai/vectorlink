{mkShell, clang, rust-bin}:
mkShell {
  buildInputs = [
    clang
    (rust-bin.nightly.latest.default.override {
      extensions = [ "rust-src" "rust-analyzer" ];
    })
  ];

  shellHook = ''
export RUSTFLAGS="-C target-feature=+avx2,+f16c,+fma"
'';
}
