FROM centos:latest

RUN yum install python3 -y

COPY SalaryData.csv /
COPY model.py /
COPY predict.py /

RUN pip3 install scikit-learn matplotlib pandas seaborn joblib   
