{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-25.05";
    flakelight = {
      url = "github:nix-community/flakelight";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    mill-scale = {
      url = "git+https://codeberg.org/icewind/mill-scale.git";
      inputs.flakelight.follows = "flakelight";
    };
  };
  outputs = {mill-scale, ...}:
    mill-scale ./. {
      crossTargets = [
        "x86_64-unknown-linux-musl"
        "i686-unknown-linux-musl"
        "armv7-unknown-linux-musleabihf"
        "aarch64-unknown-linux-musl"
        "x86_64-pc-windows-gnu"
      ];

      extraPaths = [./test_data ./examples ./benches ./tests];
      withOverlays = [(import ./nix/overlay.nix)];
      packages = {
        demostf-parser = pkgs: pkgs.demostf-parser;
        demostf-parser-codegen-events = pkgs: pkgs.demostf-parser-codegen-events;
        demostf-parser-codegen-props = pkgs: pkgs.demostf-parser-codegen-props;
        demostf-parser-schema = pkgs: pkgs.demostf-parser-schema;
      };

      tools = pkgs: with pkgs; [bacon cargo-insta];
    };
}
