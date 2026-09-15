{ pkgs, lib, ... }:

let
  # override with
  #   devenv shell -O languages.python.package:pkg python312
  py = "313";
in
{
  name = "nt2dev";

  languages.python = {
    enable = true;
    package = pkgs."python${py}";

    venv = {
      enable = true;
      requirements = ''
        ipykernel
        jupyterlab
        pytest
        -e .
      '';
    };
  };

  # https://devenv.sh/packages/
  packages = with pkgs; [
    black
    pyright
    taplo
    vscode-langservers-extracted
    zlib
  ];

  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
    pkgs.stdenv.cc.cc
    pkgs.zlib
  ];

  enterShell = ''
    echo "nt2dev devenv activated: $DEVENV_STATE/venv/bin/python"
  '';
}
