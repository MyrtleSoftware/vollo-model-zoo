{
  description = "Vollo Model Zoo: development shell";

  # Unstable for a uv in step with the one that writes uv.lock; flake.lock pins
  # the exact revision.
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
      # NoCC on purpose: tests/test_approx.py builds a C++ extension that loads
      # next to PyPI torch, i.e. against the host's libstdc++, so it has to be
      # compiled by the host's C++ compiler. A nix gcc newer than the host's
      # emits symbols that libstdc++ lacks (`CXXABI_1.3.15 not found`).
      devShells.${system}.default = pkgs.mkShellNoCC {
        packages = [
          pkgs.git # pre-commit fetches its codespell hook with it
          pkgs.ninja # torch.utils.cpp_extension builds test_approx.py's extension with it
          pkgs.uv
        ];

        # uv's own Python builds, as without nix: the host's CUDA driver and
        # libstdc++ stay visible to PyPI's torch.
        env.UV_PYTHON_PREFERENCE = "only-managed";
      };

      formatter.${system} = pkgs.nixfmt;
    };
}
