#===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================
    
from sparkx.Jetscape import Jetscape
from sparkx.JetAnalysis import JetAnalysis
import pytest
import os
import csv

TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), 'test_files')
TEST_JETSCAPE_DAT = os.path.join(TEST_FILES_DIR, 'test_jetscape.dat')
TEST_JET_FINDING = os.path.join(TEST_FILES_DIR, 'test_jet_finding.csv')

@pytest.fixture
def jet_analysis_instance():
    return JetAnalysis()

@pytest.fixture
def test_data():
    return Jetscape(TEST_JETSCAPE_DAT).particle_objects_list()

def test_jet_analysis(jet_analysis_instance,test_data,tmp_path):
    jet_analysis_instance.perform_jet_finding(test_data, jet_R=0.4, 
                                              jet_eta_range=(-2.,2.),
                                              jet_pt_range=(10.,None),
                                              output_filename=tmp_path/"test_jet.csv"
                                              )
    
    # Read the content of the test jet finding file
    test_jet_finding_content = []
    with open(TEST_JET_FINDING, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            test_jet_finding_content.append(row)

    # Read the content of the generated file in tmp_path
    generated_jet_finding_content = []
    with open(tmp_path/"test_jet.csv", 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            generated_jet_finding_content.append(row)

    # Compare the two contents
    assert test_jet_finding_content == generated_jet_finding_content

def test_read_jet_data(jet_analysis_instance, tmp_path):
    # Define test data
    test_jet_data = [
        [0, 10.0, 1.2, 3.4, 10, 10, 100.0, 1],
        [1, 20.0, 2.3, 4.5, 11, 11, 200.0, 1],
        [0, 15.0, 1.5, 3.7, 9, 9, 150.0, 2],
        [1, 25.0, 2.7, 4.9, 12, 12, 250.0, 2]
    ]

    # Write test data to a CSV file
    test_file_path = tmp_path / "test_jet_data.csv"
    with open(test_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(test_jet_data)

    # Call the read_jet_data function
    jet_analysis_instance.read_jet_data(test_file_path)

    # Compare jet data in jet_analysis_instance.jet_data_ with the test_jet_data
    for i, event_jets in enumerate(jet_analysis_instance.jet_data_):
        if i == 0:
            assert event_jets == [test_jet_data[i],test_jet_data[i+1]]
        if i == 1:
            assert event_jets == [test_jet_data[i+1],test_jet_data[i+2]]

def test_get_jets(jet_analysis_instance):
    # Define some sample jet data
    sample_jet_data = [
        [[0, 10.0, 1.2, 3.4, 10, 10, 100.0, 1], [1, 20.0, 2.3, 4.5, 11, 11, 200.0, 1]],
        [[0, 15.0, 1.5, 3.7, 9, 9, 150.0, 2], [1, 25.0, 2.7, 4.9, 12, 12, 250.0, 2]]
    ]

    # Set the jet_data_ attribute of the JetAnalysis instance
    jet_analysis_instance.jet_data_ = sample_jet_data

    # Call the get_jets method
    jets = jet_analysis_instance.get_jets()

    # Check if the extracted jets match the expected jets
    expected_jets = [[0, 10.0, 1.2, 3.4, 10, 10, 100.0, 1], [0, 15.0, 1.5, 3.7, 9, 9, 150.0, 2]]
    assert jets == expected_jets

def test_get_associated_particles(jet_analysis_instance):
    # Define some sample jet data
    sample_jet_data = [
        [[0, 10.0, 1.2, 3.4, 10, 10, 100.0, 1], [1, 20.0, 2.3, 4.5, 11, 11, 200.0, 1]],
        [[0, 15.0, 1.5, 3.7, 9, 9, 150.0, 2], [1, 25.0, 2.7, 4.9, 12, 12, 250.0, 2]]
    ]

    # Set the jet_data_ attribute of the JetAnalysis instance
    jet_analysis_instance.jet_data_ = sample_jet_data

    # Call the get_associated_particles method
    associated_particles = jet_analysis_instance.get_associated_particles()

    # Check if the extracted associated particles match the expected associated particles for each jet
    expected_associated_particles = [
        [[1, 20.0, 2.3, 4.5, 11, 11, 200.0, 1]],
        [[1, 25.0, 2.7, 4.9, 12, 12, 250.0, 2]]
    ]
    assert associated_particles == expected_associated_particles
