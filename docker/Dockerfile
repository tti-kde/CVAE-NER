FROM nvcr.io/nvidia/pytorch:24.05-py3

ENV UID 1000001
ENV USER_NAME user
ENV HF_HOME ~/data/.cache/huggingface

# install essential softwares
RUN rm -R /etc/apt/sources.list.d
RUN apt update && apt install -y vim bash git ssh sudo language-pack-en tmux bc libcairo2-dev pkg-config python3-dev
RUN python -m pip install --upgrade pip

# Install python packages
# Hint: pip
# command
RUN pip install --upgrade pip
RUN pip install tqdm black pycairo
RUN pip install optuna
RUN pip install -q -U git+https://github.com/huggingface/transformers.git
RUN pip install -U git+https://github.com/huggingface/peft.git
RUN pip install -U bitsandbytes
RUN pip install pydantic==1.10.11
RUN pip install overrides
RUN pip install requests
RUN pip install numpy
RUN pip install wandb
RUN pip install pandas
RUN pip install bioc datasets openpyxl
RUN pip install seaborn
RUN pip install adjustText

# Omajinai
RUN update-locale LANG=en_US.UTF-8

WORKDIR /workspace
RUN yes | adduser --uid $UID --force-badname --shell /bin/bash --disabled-password ${USER_NAME} -q && echo ${USER_NAME}:${USER_NAME} | chpasswd
RUN ln -s /workspace /home/"${USER_NAME}"/workspace && chown ${USER_NAME}:${USER_NAME} /workspace && chmod 755 /workspace

USER ${USER_NAME}
