{
  inputs = {
    nixpkgs.url = "github:nixOS/nixpkgs?ref=nixpkgs-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs = {
        nixpkgs.follows = "nixpkgs";
      };
    };
  };

  outputs = { self, nixpkgs, rust-overlay }:(
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      nixpkgsFor = forAllSystems (system:
        (import nixpkgs) {
          inherit system;
          overlays = [
            (import rust-overlay)
          ];
        }
      ); in
      {
        devShells = forAllSystems (system :
          let pkgs = nixpkgsFor.${system};in
          {
            default = pkgs.callPackage ./shell.nix {};
          });
      }
  );
}
