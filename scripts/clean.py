#!/usr/bin/env python

import subprocess
import os


def main():
    script_path = os.path.join(os.path.dirname(__file__), "clean.sh")

    # Verifica se o script existe antes de executá-lo
    if not os.path.exists(script_path):
        print(f"Erro: Script {script_path} não encontrado.")
        return

    subprocess.run(["bash", script_path], check=True)


if __name__ == "__main__":
    main()
