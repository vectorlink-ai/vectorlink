{mkShell, clang, rust-bin, openssl, pkg-config}:
mkShell {
  buildInputs = [
    clang
    pkg-config
    openssl
    (rust-bin.nightly.latest.default.override {
      extensions = [ "rust-src" "rust-analyzer" ];
    })
  ];

  shellHook = ''
export RUSTFLAGS="-C target-feature=+avx2,+f16c,+fma,+aes,+sse2"
'';
}
