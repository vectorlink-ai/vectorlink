{mkShell, clang, rust-bin}:
mkShell {
  buildInputs = [
    clang
    (rust-bin.nightly.latest.default.override {
      extensions = [ "rust-src" "rust-analyzer" ];
    })
  ];
}
