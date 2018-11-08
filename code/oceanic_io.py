from amuse.units import units
import numpy as np
import dill
from amuse.units import units
import os

class meta_array(np.ndarray):
    """Array with metadata."""

    def __new__(cls, array, dtype=None, order=None, **kwargs):
        obj = np.asarray(array, dtype=dtype, order=order).view(cls)
        obj.meta = kwargs
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.meta = getattr(obj, 'metadata', None)

class snapshot_reader(object):
    def __init__(self, options_reader, galaxy_code):
        options_reader.set_options(self)
        self.frames = meta_array([])
        if not self.axisymmetric:
            self.frames.meta['ss_id'] = galaxy_code.chosen_id
        self.frames.meta['simulation_directory'] = galaxy_code.simulation_directory

    def process_snapshot(self, system, galaxy_code, com, i, time, final=None):
        f = self._grab_frame_(system, galaxy_code, com, time)
        self.frames = meta_array(np.append(self.frames, f), **self.frames.meta)
        # self.frames.append(self._grab_frame_(system, galaxy_code, time))
        if np.mod(i, self.write_frequency) == 0:
            fout = self.output_directory+'/cluster_snapshots.p'
            dill.dump(self.frames, open(fout, 'wb'))

    def finish_sim(self):
        fout = self.output_directory+'/cluster_snapshots.p'
        dill.dump(self.frames, open(fout, 'wb'))

    def _grab_frame_(self, system, galaxy_code, com, time):
        x = system.particles.x.value_in(units.parsec)
        y = system.particles.y.value_in(units.parsec)
        z = system.particles.z.value_in(units.parsec)

        vx = system.particles.vx.value_in(units.kms)
        vy = system.particles.vy.value_in(units.kms)
        vz = system.particles.vz.value_in(units.kms)

        mass = system.particles.mass.value_in(units.MSun)

        position = np.transpose([x, y, z])
        velocity = np.transpose([vx, vy, vz])

        time = time.value_in(units.Myr)

        frame = {'time': time,
                 'position': position,
                 'velocity': velocity,
                 'mass': mass,
                 'com': com}

        if not self.axisymmetric:
            chosen_position = galaxy_code.chosen_evolved_position * 1000.0
            chosen_velocity = galaxy_code.chosen_evolved_velocity
            frame['chosen_position'] = chosen_position
            frame['chosen_velocity'] = chosen_velocity

        return frame

def dump_interface(interface, directory_out='interface'):
    directory_out = interface.output_directory + '/' + directory_out
    if not os.path.isdir(directory_out):
        os.makedirs(directory_out)

    # delete stuff we won't ever need again
    del interface.first_snapshot
    del interface.snapshots
    try: # for backwards compatibility
        del interface.star_snapshots
    except:
        pass
    del interface.first_ag

    # now dump grid piecemeal
    dill.dump(interface.grid.evolved_grid, open(directory_out+'/evolved_grid', 'wb'))
    dill.dump(interface.grid.init_grid, open(directory_out+'/init_grid', 'wb'))
    del interface.grid.evolved_grid
    del interface.grid.init_grid

    dill.dump(interface.grid.grid_accx_interpolators, open(directory_out+'/grid_accx_interpolators', 'wb'))
    dill.dump(interface.grid.grid_accy_interpolators, open(directory_out+'/grid_accy_interpolators', 'wb'))
    dill.dump(interface.grid.grid_accz_interpolators, open(directory_out+'/grid_accz_interpolators', 'wb'))
    del interface.grid.grid_accx_interpolators
    del interface.grid.grid_accy_interpolators
    del interface.grid.grid_accz_interpolators

    dill.dump( interface.grid.snapshot_acceleration_x, open(directory_out+'/snapshot_acceleration_x.npz', 'wb'))
    dill.dump(interface.grid.snapshot_acceleration_y, open(directory_out+'/snapshot_acceleration_y.npz', 'wb'))
    dill.dump(interface.grid.snapshot_acceleration_z, open(directory_out+'/snapshot_acceleration_z.npz', 'wb'))
    del interface.grid.snapshot_acceleration_x
    del interface.grid.snapshot_acceleration_y
    del interface.grid.snapshot_acceleration_z

    del interface.grid.evolved_acceleration_x
    del interface.grid.evolved_acceleration_y
    del interface.grid.evolved_acceleration_z

    dill.dump(interface, open(directory_out+'/interface', 'wb'))


def load_interface(directory='interface', skinny=True):
    interface = dill.load(open(directory+'/interface', 'rb'))

    interface.grid.snapshot_acceleration_x = np.load(directory+'/snapshot_acceleration_x.npz')
    interface.grid.snapshot_acceleration_y = np.load(directory+'/snapshot_acceleration_y.npz')
    interface.grid.snapshot_acceleration_z = np.load(directory+'/snapshot_acceleration_z.npz')

    interface.grid.evolved_grid = dill.load(open(directory+'/evolved_grid', 'rb'))
    interface.grid.init_grid = dill.load(open(directory+'/init_grid', 'rb'))

    if skinny:
        interface.ncpu = 1
        interface._init_acceleration_grid_interpolators_()
    else:
        interface.grid.grid_accx_interpolators = dill.load(open(directory+'/grid_accx_interpolators', 'rb'))
        interface.grid.grid_accy_interpolators = dill.load(open(directory+'/grid_accy_interpolators', 'rb'))
        interface.grid.grid_accz_interpolators = dill.load(open(directory+'/grid_accz_interpolators', 'rb'))
        interface._init_acceleration_pool_()
    interface.evolve_model(0 | units.Myr)

    # if filein is None:
    #     if skinny:
    #         filein = 'interface_skinny.p'
    #     else:
    #         filein = 'interface.p'
    # interface = dill.load(open(filein, 'rb'))
    # from oceanic.gizmo_interface import acc_wrapper, run_worker_x, run_worker_y, run_worker_z
    # if skinny:
    #     interface._init_acceleration_grid_interpolators_()
    # else:
    #     interface._init_acceleration_pool_()

    return interface
