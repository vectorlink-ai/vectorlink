{pkgs, mkShell}:
let
  pythonVenvShellHook = ''
    # Setup the virtual environment if it doesn't already exist.
    VENV=.venv
    if test ! -d $VENV; then
      virtualenv $VENV
    fi
    source ./$VENV/bin/activate
    export PYTHONPATH=`pwd`/$VENV/${pkgs.python3.sitePackages}/:$PYTHONPATH
  '';
in
mkShell {
  buildInputs = with pkgs; [
    clang
    pkg-config
    maturin
    openssl
    python3 # For pyo3/maturin, which is used by hnsw_redux-python
    (python3.withPackages (pythonPkgs: with pythonPkgs; [
      # Note that even if Python packages like PyTorch or Tensorflow are added,
      # they will be reinstalled when running `pip -r requirements.txt` because
      # virtualenv is used below in the shellHook. Fkn virtualenv :/
      ipython
      pip
      setuptools
      virtualenvwrapper
      wheel
    ]))
    (rust-bin.nightly."2025-01-23".default.override {
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
  '' + pythonVenvShellHook else if pkgs.system == "aarch64-linux" then ''
    export RUSTFLAGS="-C target-feature=+neon"
  '' + pythonVenvShellHook else if pkgs.system == "aarch64-darwin" then ''
    export RUSTFLAGS="-C target-feature=+neon"
  '' + pythonVenvShellHook else throw "Unknown system: ${pkgs.system}";
}
