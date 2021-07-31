FROM signate/runtime-jpx:2021.03
COPY requirements.txt /workspace/requirements.txt
WORKDIR /workspace
RUN pip install -U pip
RUN pip install TA-Lib
RUN pip install -r /workspace/requirements.txt