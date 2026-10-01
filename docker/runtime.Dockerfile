# Runtime image: builder toolchain + PyTorch 2.14 wheel + atomistic-ML stack,
# with the multi-user entrypoint (see docs/MULTIUSER.md).
#
# Build from the repository root:  scripts/build_images.sh runtime

ARG BUILDER_IMAGE=iapetus/builder:cuda11.8-py312
FROM ${BUILDER_IMAGE}

ARG TORCH_WHEEL=dist/torch-2.14.0.post2-cp312-cp312-linux_x86_64.whl

USER root

COPY ${TORCH_WHEEL} /tmp/wheels/
RUN uv pip install --python /opt/venv312 --no-deps /tmp/wheels/*.whl && rm -rf /tmp/wheels

COPY docker/requirements.txt /tmp/requirements.txt
RUN uv pip install --python /opt/venv312 --no-deps -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY third_party/pytorch_scatter /tmp/pytorch_scatter
COPY scripts/build_torch_scatter.sh /tmp/build_torch_scatter.sh
RUN /tmp/build_torch_scatter.sh /tmp/pytorch_scatter && rm -rf /tmp/pytorch_scatter /tmp/build_torch_scatter.sh

COPY third_party/openequivariance /tmp/openequivariance
COPY scripts/build_openequivariance.sh /tmp/build_openequivariance.sh
RUN /tmp/build_openequivariance.sh /tmp/openequivariance && rm -rf /tmp/openequivariance /tmp/build_openequivariance.sh

COPY scripts/build_metatomic_torch.sh /tmp/build_metatomic_torch.sh
RUN /tmp/build_metatomic_torch.sh && rm /tmp/build_metatomic_torch.sh

# Forces use_libuv=False for TCPStore (PyTorch is built without libuv)
COPY docker/sitecustomize.py /opt/venv312/lib/python3.12/site-packages/sitecustomize.py

ENV USE_LIBUV=0 \
    CMAKE_PREFIX_PATH="/usr/local:/opt/venv312/lib/python3.12/site-packages/torch/share/cmake:/opt/venv312/lib/python3.12/site-packages/metatensor/lib/cmake:/opt/venv312/lib/python3.12/site-packages/metatensor_torch/torch-2.14/lib/cmake"

RUN (getent group render >/dev/null || groupadd -g 991 render) && \
    (getent group kvm >/dev/null || groupadd -g 992 kvm) && \
    echo 'export PATH="/opt/venv312/bin:$PATH"' >> /etc/bash.bashrc && \
    echo 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true' >> /etc/bash.bashrc && \
    echo 'export USE_LIBUV=0' >> /etc/bash.bashrc && \
    echo "export CMAKE_PREFIX_PATH=\"${CMAKE_PREFIX_PATH}:\${CMAKE_PREFIX_PATH:-}\"" >> /etc/bash.bashrc && \
    mkdir -p /workspace && chmod 1777 /tmp

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
