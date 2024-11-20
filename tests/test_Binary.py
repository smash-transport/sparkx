# ===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

import struct
import pytest
import random
import tempfile
import sparkx.Oscar as Oscar
import numpy as np


def writeHeader(file):
    file.write(struct.pack('4s', b'SMSH'))
    file.write(struct.pack('H', 9))
    file.write(struct.pack('H', 1))
    smash_version = 'SMASH-3.1'
    file.write(struct.pack('I', len(smash_version)))
    file.write(smash_version.encode('utf-8'))
    return

def writeParticle(file, particle):
    for element in particle:
        if isinstance(element, int):
            file.write(struct.pack('i', element))
        else:
            file.write(struct.pack('d', element))
    return

def writeParticleBlock(file, particles):
    file.write(struct.pack('1s', b'p'))
    file.write(struct.pack('I', len(particles)))
    for particle in particles:
        writeParticle(file, particle)
    return

def writeEndBlock(file, event_number, impact_parameter, empty):
    empty_flag = b'0' if empty else b'1'
    file.write(struct.pack('1s', b'f'))
    file.write(struct.pack('I', event_number))
    file.write(struct.pack('d', impact_parameter))
    file.write(struct.pack('1s', empty_flag))
    return

def generateParticles(num_particles):
    particles = []
    for i in range(num_particles):
        t = random.uniform(0.0, 100.0)
        x = random.uniform(-10.0, 10.0)
        y = random.uniform(-10.0, 10.0)
        z = random.uniform(-10.0, 10.0)
        mass = random.choice([0.938, 0.139, 1.875, 0.497])
        p0 = random.uniform(0.5, 10.0)
        px = random.uniform(-5.0, 5.0)
        py = random.uniform(-5.0, 5.0)
        pz = random.uniform(-5.0, 5.0)
        pdgid = random.choice([211, -211, 2212, -2212, 311, -311])
        id = i
        charge = random.choice([-1, 0, 1])
        ncoll = random.randint(0, 10)
        form_time = random.uniform(0.0, 1.0)
        xsecfac = random.uniform(0.0, 1.0)
        proc_id_origin = random.randint(100, 200)
        proc_type_origin = random.randint(200, 300)
        time_last_coll = random.uniform(0.0, t)
        pdg_mother1 = random.choice([111, 211, 311])
        pdg_mother2 = random.choice([111, 211, 311])
        baryon_number = random.choice([0, 1])
        strangeness = random.choice([-1, 0, 1])

        particle = [
            t, x, y, z, mass, p0, px, py, pz, pdgid, id, charge, ncoll,
            form_time, xsecfac, proc_id_origin, proc_type_origin,
            time_last_coll, pdg_mother1, pdg_mother2, baryon_number, strangeness
        ]
        particles.append(particle)
    return particles

@pytest.mark.parametrize("events", [
    ([generateParticles(5), generateParticles(10)]),
    ([generateParticles(1), generateParticles(4), generateParticles(7)]),
])
def test_loading_binary(events):
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as temp_file:
        # Write data to the binary file
        with open(temp_file.name, "wb") as file:
            writeHeader(file)
            for i, event in enumerate(events):
                writeParticleBlock(file, event)
                writeEndBlock(file, event_number=i, impact_parameter=0.0, empty=False)
        
        # Use SparkX to read the data back
        oscar = Oscar(temp_file.name)
        read_events = oscar.particle_list()

    # Validate the data read matches the original
    for original_event, read_event in zip(events, read_events):
        assert(original_event == read_event)
