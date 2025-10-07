{
  pkgs ? import <nixpkgs> { },
  py ? "312",
}:

pkgs.mkShell {
  name = "nt2dev";
  nativeBuildInputs = with pkgs; [
    pkgs."python${py}"
    pkgs."python${py}Packages".pip
    black
    pyright
    taplo
    vscode-langservers-extracted
    zlib
  ];
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc
    pkgs.zlib
  ];

  shellHook = ''
    if [ ! -d ".venv" ]; then
      python3 -m venv .venv
      source .venv/bin/activate
      pip3 install -r requirements.txt
      pip3 install pytest
      pip3 install -e .
    else
      source .venv/bin/activate
    fi
    echo "nt2dev nix-shell activated: $(which python)"
  '';
}
