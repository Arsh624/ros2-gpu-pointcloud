FROM nvidia/cuda:12.6.3-devel-ubuntu24.04@sha256:392c0df7b577ecae17a17f6ba7f2009c217bb4422f8431c053ae9af61a8c148a

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
SHELL ["/bin/bash", "-c"]

# Layer 1: base tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg lsb-release build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: ROS 2 apt repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
        > /etc/apt/sources.list.d/ros2.list

# Layer 3: ROS 2 Jazzy
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-base ros-jazzy-rmw-zenoh-cpp ros-jazzy-rmw-cyclonedds-cpp \
        ros-jazzy-ament-cmake-gtest python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Layer 4: profiling tools (kept last so earlier layers stay cached)
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nsight-systems-12-6 \
    && rm -rf /var/lib/apt/lists/*

# Layer 5: PCL, used only by the optional PCL baseline in the benchmark (WITH_PCL=ON)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpcl-dev \
    && rm -rf /var/lib/apt/lists/*

# Layer 6: Fast DDS, Jazzy's default RMW. ros-base installs without it under --no-install-recommends,
# and the node then silently falls back to Cyclone DDS. Separate layer so the ones above stay cached.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-rmw-fastrtps-cpp \
    && rm -rf /var/lib/apt/lists/*

RUN printf '%s\n' \
        '#!/bin/bash' \
        'set -e' \
        'source /opt/ros/jazzy/setup.bash' \
        'if [ -f /ws/install/setup.bash ]; then source /ws/install/setup.bash; fi' \
        'exec "$@"' \
        > /ros_entrypoint.sh \
    && chmod +x /ros_entrypoint.sh \
    && echo 'source /opt/ros/jazzy/setup.bash' >> /etc/bash.bashrc

WORKDIR /ws
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
