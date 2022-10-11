FROM docker.elastic.co/elasticsearch/elasticsearch:7.16.2
RUN elasticsearch-plugin install --batch https://github.com/alexklibisz/elastiknn/releases/download/7.16.2.0/elastiknn-7.16.2.0.zip

WORKDIR /usr/share/elasticsearch/

HEALTHCHECK --interval=10s --timeout=30s --start-period=10s CMD curl --fail http://localhost:9200 || exit 1
