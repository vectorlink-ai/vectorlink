{pkgs, mkShell}:
let
  pythonVenvShellHook = ''
    # Setup a fresh virtualenv when `direnv reload` is executed.  This is done
    # to keep the project working after a `nix-collect-garbage` removes the
    # pointee entry in the nix store. When this happens, the pointer
    # (e.g. `$VENV/bin/python`) becomes stale and breaks.

    VENV=.venv
    # Test for the existence of:
    #   * The existence of the $VENV directory
    #   * The *POINTEE* of the `$VENV/bin/python` symlink:
    if [[ ! (-d $VENV  &&  -f $VENV/bin/python) ]]; then
      rm -rf $VENV  # Remove any existing virtualenv
      virtualenv $VENV  # Setup a fresh virtualenv
      source ./$VENV/bin/activate
      export PYTHONPATH=`pwd`/$VENV/${pkgs.python3.sitePackages}/:$PYTHONPATH
    fi
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
      /*
      (datafusion.overridePythonAttrs rec {
        version = "43.1.0";
        doCheck = false;
        src = fetchFromGitHub {
          name = "datafusion-source";
          owner = "apache";
          repo = "datafusion-python";
          tag = version;
          hash = "sha256-a/6x+9xAHgZmTnmrqnI9264fbgWykUkutMjcZHZdMPE=";
        };

        cargoDeps = rustPlatform.fetchCargoVendor {
          name = "datafusion-cargo-deps";
          inherit src;
          hash = "sha256-KkU8cN74Vfh3kp1O9cvBqevxnLXnKNA+J4sttNgf5S0=";
        };

      })
*/
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
  # add manylinux1 to ld library path to allow pip packages to find what they need without rpath patching.
  LD_LIBRARY_PATH=pkgs.lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;

  shellHook = (if pkgs.system == "x86_64-linux" then ''
    export RUSTFLAGS="-C target-feature=+avx2,+f16c,+fma,+aes,+sse2"
  '' else if pkgs.system == "aarch64-linux" then ''
    export RUSTFLAGS="-C target-feature=+neon"
  '' else if pkgs.system == "aarch64-darwin" then ''
    export RUSTFLAGS="-C target-feature=+neon"
  '' else throw "Unknown system: ${pkgs.system}"
  ) + pythonVenvShellHook;
}
