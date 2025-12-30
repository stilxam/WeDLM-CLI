{
  description = "Flake for WeDLM CLI";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11"; 
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:

    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        python = pkgs.python313;

        wedlm = pkgs.python313Packages.buildPythonPackage {
          pname = "wedlm";
          version = "0.1.0";
          pyproject = true;

          src = pkgs.fetchFromGitHub {
            owner = "Tencent";
            repo = "WeDLM";
            rev = "3624ddc9e9775ce162f6f60158745f9c79437582";
            hash = "sha256-S02+spjXtUKilmPj8SCFlKQ4MgLOI56ugNFc7vbHMqk=";
          };

          nativeBuildInputs = with pkgs.python313Packages; [
            setuptools
            wheel
          ];

          propagatedBuildInputs = with pkgs.python313Packages; [
            torch
            transformers
            triton
            flash-attn
            xxhash
            numpy
            tqdm
            safetensors
            flask
          ];

          doCheck = false; 
        };

        mainPythonPackages = ps: with ps; [
          cython
          pytest
          black
          isort
          flake8
          huggingface-hub 
          rich
	  pylatexenc
          wedlm
        ];

        pythonEnv = python.withPackages mainPythonPackages;

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.zed-editor
            pkgs.git-lfs
          ];

          shellHook = ''
            # Set CUDA paths
            export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit}
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc pkgs.cudaPackages.cudatoolkit ]}:$LD_LIBRARY_PATH
            
            # Model management
            MODEL_NAME="tencent/WeDLM-8B-Instruct"
            LOCAL_DIR="WeDLM-8B-Instruct"

            if [ ! -d "$LOCAL_DIR" ]; then
              echo "Model folder '$LOCAL_DIR' not found."
              echo "Downloading $MODEL_NAME"
              huggingface-cli download $MODEL_NAME --local-dir $LOCAL_DIR
            else
              echo "Model folder '$LOCAL_DIR' already exists."
            fi
          '';
        };
      });
}
