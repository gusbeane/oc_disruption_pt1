import numpy as np
from amuse.units import units
from amuse.couple.bridge import Bridge
from tqdm import tqdm

import sys

from oceanic.oc_code import oc_code
from oceanic.options import options_reader

from oceanic.gizmo_interface import gizmo_interface
from oceanic.oceanic_io import snapshot_reader


def evolve_cluster_in_galaxy(options_file):
    opt = options_reader(options_file)

    timestep = opt.options['timestep']  # in Myr
    tend = opt.options['tend']  # in Myr
    times = np.arange(0.0, tend, timestep) | units.Myr

    cluster = oc_code(opt)
    cluster_code = cluster.code
    galaxy_code = gizmo_interface(opt)

    snap_reader = snapshot_reader(opt, galaxy_code)

    stars = cluster_code.particles.copy()

    if opt.options['axisymmetric']:
        import astropy.units as u
        import gala.dynamics as gd
        import gala.potential as gp

        pos = [opt.options['axi_Rinit'], 0, 0] * u.kpc
        vel = [0, 0, 0] * u.km/u.s
        mw = gp.MilkyWayPotential()
        phase = gd.PhaseSpacePosition(pos, vel)
        vc = mw.circular_velocity(phase).to_value(u.km/u.s) | units.kms

        stars = cluster_code.particles.copy()
        stars.x += opt.options['axi_Rinit'] | units.kpc
        stars.vy += vc
        stars.z += opt.options['axi_zinit'] | units.kpc

    channel = stars.new_channel_to(cluster_code.particles)
    channel.copy_attributes(["x", "y", "z", "vx", "vy", "vz"])

    system = Bridge(timestep=timestep, use_threading=False)
    system.add_system(cluster_code, (galaxy_code,))
    system.add_system(galaxy_code)

    converter = cluster_code.unit_converter

    for i, t in enumerate(tqdm(times)):
        system.evolve_model(t, timestep=timestep | units.Myr)
        if not opt.options['axisymmetric']:
            cluster.clean_ejections(system)

        bound = system.particles.bound_subset(unit_converter=converter)
        bound_com = bound.center_of_mass().value_in(units.kpc)

        if not opt.options['axisymmetric']:
            galaxy_code.evolve_grid(bound_com)

        snap_reader.process_snapshot(system, galaxy_code, bound_com, i, t)

    snap_reader.finish_sim()

    cluster_code.stop()


if __name__ == '__main__':
    evolve_cluster_in_galaxy(sys.argv[1])
