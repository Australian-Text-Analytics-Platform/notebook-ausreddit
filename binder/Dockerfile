# Note that there must be a tag
FROM jupyter/scipy-notebook:cf6258237ff9

ADD https://data.ldaca.edu.au/api/object/arcp%3A%2F%2Fname%2Chdl10.25912~RDF_1754373503052%2Ftopics_crate%2F.zip
RUN unzip -n 'arcp:%2F%2Fname,hdl10.25912~RDF_1754373503052%2Ftopics_crate%2F.zip'

ADD https://data.ldaca.edu.au/api/object/arcp%3A%2F%2Fname%2Chdl10.25912~RDF_1754373503052%2F1grams_crate%2F.zip
RUN unzip -n 'arcp:%2F%2Fname,hdl10.25912~RDF_1754373503052%2F1grams_crate%2F.zip'

ADD https://data.ldaca.edu.au/api/object/arcp%3A%2F%2Fname%2Chdl10.25912~RDF_1754373503052%2Fdomains_crate%2F.zip
RUN unzip -n 'arcp:%2F%2Fname,hdl10.25912~RDF_1754373503052%2Fdomains_crate%2F.zip'

ADD https://data.ldaca.edu.au/api/object/arcp%3A%2F%2Fname%2Chdl10.25912~RDF_1754373503052%2F3grams_crate%2F.zip
RUN unzip -n 'arcp:%2F%2Fname,hdl10.25912~RDF_1754373503052%2F3grams_crate%2F.zip'

ADD https://data.ldaca.edu.au/api/object/arcp%3A%2F%2Fname%2Chdl10.25912~RDF_1754373503052%2Femotions_crate%2F.zip
RUN unzip -n 'arcp:%2F%2Fname,hdl10.25912~RDF_1754373503052%2Femotions_crate%2F.zip'


RUN rm ro-crate-metadata.json
RUN rm ro-crate-preview.html
RUN rm */*.csv
RUN rm *.zip
rmdir 1grams_crate/ 3grams_crate/ emotions_crate/ topics_crate/
RUN mv */*.parquet