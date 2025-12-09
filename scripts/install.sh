#!/bin/bash

# Variáveis para controle
docker_cmd="./docker/create_images.sh"
pip_cmd="sudo pip install ."

# Verifica as flags
while getopts "di" opt; do
  case ${opt} in
    d )
      $docker_cmd
      exit 0
      ;;
    i )
      $pip_cmd
      exit 0
      ;;
    * )
      echo "Uso: $0 [-d] [-i]"
      exit 1
      ;;
  esac
done

# Se nenhuma flag for passada, executa ambos os comandos
$docker_cmd
$pip_cmd
