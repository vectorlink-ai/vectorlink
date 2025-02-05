{pkgs, mkShell}:
let
  pythonVenvShellHook = ''
cd hnsw_redux-python
uv sync --no-install-project
source .venv/bin/activate
cd ..
'';
in
mkShell {
  buildInputs = with pkgs; [
    clang
    pkg-config
    maturin
    openssl
    python3 # For pyo3/maturin, which is used by hnsw_redux-python
    uv
    (rust-bin.nightly."2025-01-23".default.override {
      extensions = [ "rustfmt" "rust-src" "rust-analyzer" ];
      targets = [
        "aarch64-apple-darwin"
        "aarch64-unknown-linux-gnu"
        "x86_64-unknown-linux-gnu"
      ];
    })
  ];
  # add manylinux1 to ld library path to allow pip packages to find what they need without rpath patching.

  LD_LIBRARY_PATH = if pkgs.stdenv.isDarwin then
    ""
  else
    pkgs.lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;

  shellHook = (if pkgs.system == "x86_64-linux" then ''
    export RUSTFLAGS="-C target-feature=+avx2,+f16c,+fma,+aes,+sse2"
  '' else if pkgs.system == "aarch64-linux" then ''
    export RUSTFLAGS="-C target-feature=+neon"
  '' else if pkgs.system == "aarch64-darwin" then ''
    export RUSTFLAGS="-C target-feature=+neon"
  '' else throw "Unknown system: ${pkgs.system}"
  ) + pythonVenvShellHook;
}
