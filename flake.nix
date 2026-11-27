{
  description = "Vollo Model Zoo: development shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      # The Vollo SDK's wheels are x86_64 Linux only.
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      # Python and its packages come from uv.lock, not nix: this shell supplies
      # only what uv can't.
      #
      # NoCC on purpose: tests build C++ extensions that load next to PyPI
      # torch, i.e. against the host's libstdc++, so it has to be compiled by
      # the host's C++ compiler.
      devShells.${system}.default = pkgs.mkShellNoCC {
        packages = [
          pkgs.git # pre-commit fetches its codespell hook with it
          pkgs.ninja # torch.utils.cpp_extension builds extension with it
          pkgs.uv
        ];

        # uv's own Python builds, as without nix: the host's CUDA driver and
        # libstdc++ stay visible to PyPI's torch.
        env.UV_PYTHON_PREFERENCE = "only-managed";
      };

      formatter.${system} = pkgs.nixfmt;
    };
}
