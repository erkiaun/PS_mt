#!/bin/bash
# Script for testing model building with PhenotypeSeeker, using
# the Clostridium difficie azithromycin resistance dataset as an
# example.

# Download C. difficile FASTA files and inputfile for PhenotypeSeeker
echo Downloading the folder with example files ...
echo
wget http://bioinfo.ut.ee/PhenotypeSeeker/PS_modeling_example_files.tar.gz

#Unpack the downloaded folder
echo Unpacking the folder ...
echo
tar -zxvf PS_modeling_example_files.tar.gz

#Launch the "PhenotypeSeeker modeling"
echo
echo "Launching the phenotypeseeker modeling:"
echo "phenotypeseeker modeling PS_modeling_example_files/data.pheno -w"
echo

phenotypeseeker modeling PS_modeling_example_files/data.pheno 

echo "Finished!"