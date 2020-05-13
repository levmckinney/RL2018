#!/bin/bash
work_space="$(dirname $(realpath $0))"
docker_file="${work_space}/.devcontainer/Dockerfile"
container_name="rl2018-lev"
for i in "$@"
do
case $i in
  -b|--build)
  echo $docker_file
  docker build -f $docker_file -t $container_name $work_space
  ;;
  -r|--run)
  docker run -p 8888:8888 --mount type=bind,source=$work_space,target=/workspaces/RL2018 \
  --cap-add SYS_ADMIN \
  --entrypoint /bin/bash \
  $container_name \
  -c "source ~/.cargo/env && cd workspaces/RL2018 && jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root"
  ;;
esac
done
