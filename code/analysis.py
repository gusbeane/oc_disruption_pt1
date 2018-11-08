import matplotlib as mpl; mpl.use('agg')

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.animation import FuncAnimation
from amuse.units import units

import agama
import gizmo_analysis as gizmo
import numpy as np
from tqdm import tqdm

import sys
import dill

#from oceanic.gizmo_interface import gizmo_interface
from oceanic.options import options_reader
from amuse.units import units
from tqdm import tqdm

class agama_wrapper(object):
    def __init__(self, opt):
        opt.set_options(self)
        agama.setUnits(mass=1, length=1, velocity=1)

    def update_index(self, index, ss_id=None, snap=None):
        agama.setUnits(mass=1, length=1, velocity=1)
        self.current_index = index
        self.ss_id = ss_id

        if snap is not None:
            self.snap = snap
        else:
            self.snap = \
                gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'],
                                             'index', index,
                                             properties=['id', 'position',
                                                         'velocity', 'mass',
                                                         'form.scalefactor'],
                                             simulation_directory=
                                             self.simulation_directory,
                                             assign_principal_axes=True)

        if self.sim_name is None:
            head = gizmo.io.Read.read_header(snapshot_value=self.startnum,
                                             simulation_directory=
                                             self.simulation_directory)
            self.sim_name = head['simulation.name'].replace(" ", "_")

        potential_cache_file = self.cache_directory + '/potential_id'+str(index)
        potential_cache_file += '_' + self.sim_name + '_pot'
        try:
            self.potential = agama.Potential(file=potential_cache_file)
        except:
            star_position = self.snap['star'].prop('host.distance.principal')
            gas_position = self.snap['gas'].prop('host.distance.principal')
            dark_position = self.snap['dark'].prop('host.distance.principal')

            star_mass = self.snap['star']['mass']
            gas_mass = self.snap['gas']['mass']
            dark_mass = self.snap['dark']['mass']



            position = np.concatenate((star_position, gas_position))
            mass = np.concatenate((star_mass, gas_mass))

            #TODO make these user-controllable
            self.pdark = agama.Potential(type="Multipole",
                                        particles=(dark_position, dark_mass),
                                        symmetry='a', gridsizeR=20, lmax=2)
            self.pbar = agama.Potential(type="CylSpline",
                                        particles=(position, mass),
                                        gridsizer=20, gridsizez=20,
                                        mmax=0, Rmin=0.2, symmetry='a',
                                        Rmax=50, Zmin=0.02, Zmax=10)
            self.potential = agama.Potential(self.pdark, self.pbar)
            self.potential.export(potential_cache_file)

        if ss_id is not None:
            self.ss_init = True
            ss_key = np.where(self.snap['star']['id'] == self.ss_id)[0]
            self.chosen_position = self.snap['star'].prop(
                'host.distance.principal')[ss_key]
            self.chosen_velocity = self.snap['star'].prop(
                'host.velocity.principal')[ss_key]
        else:
            self.ss_init = False

        self.af = agama.ActionFinder(self.potential, interp=False)

    def update_ss(self, ss_id, position=None, velocity=None):
        self.ss_init = True
        ss_key = np.where(self.snap['star']['id'] == ss_id)[0]
        if position is not None:
            self.chosen_position = position
        else:
            self.chosen_position = self.snap['star'].prop(
                'host.distance.principal')[ss_key]
        if velocity is not None:
            self.chosen_velocity = velocity
        else:
            self.chosen_velocity = self.snap['star'].prop(
                'host.velocity.principal')[ss_key]

    def actions(self, poslist, vlist, add_ss=False, in_kpc=False):
        if not in_kpc:
            poslist = poslist.copy()/1000.0
        if add_ss:
            if not self.ss_init:
                raise Exception('need to initialize with a ss id to add \
                                 ss pos, vel')
            else:
                poslist = np.add(poslist, self.chosen_position)
                vlist = np.add(vlist, self.chosen_velocity)
        points = np.c_[poslist, vlist]
        return self.af(points) # Jr, Jz, Lz

    def ss_action(self):
        points = np.c_[self.chosen_position, self.chosen_velocity]
        return self.af(points)[0]


class snapshot_action_calculator(object):
    def __init__(self, options,
                 snapshot_file='cluster_snapshots.p', ss_id=None):
        options.set_options(self)
        self._ag_ = agama_wrapper(options)

        try:
            self.cluster = dill.load(open(snapshot_file, 'rb'))
        except:
            raise Exception('could not find snapshot file: ', snapshot_file)

        if self.axisymmetric:
            return None

        if ss_id is None:
            try:
                self.ss_id = self.cluster.meta['ss_id']
            except:
                raise Exception('cant find ss_id in cluster_snapshots⁠ \
                                    please specify ss id')
        else:
            self.ss_id = ss_id

    def snapshot_actions(self, fileout='cluster_snapshots_snap_actions.npy',
                         start=None, end=None):
        # start, end are int's describing where to start and ending
        # if both are None, will scroll through all snapshots where
        # snapshot_file spans
        #
        # If start is an int and end is None, will just do start index
        # If both are not None, will go from start to end (make sure
        # snapshot_file spans what you want actions from)

        if start is None and end is None:
            self.snapshot_indices = list(range(self.startnum,
                                               self.endnum+1))
            self.start = self.snapshot_indices[0]
            self.end = self.snapshot_indices[-1]
        elif start is not None and end is None:
            self.snapshot_indices = (start,)
            self.start = start
            self.end = end
        elif start is not None and end is not None:
            self.snapshot_indices = list(range(start, end+1))
            self.start = start
            self.end = end
        else:
            raise Exception('invalid start and end combination')

        self.first_snap = \
            gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'],
                                          'index', self.startnum,
                                         properties=['id', 'position',
                                                     'velocity', 'mass',
                                                     'form.scalefactor'],
                                         simulation_directory=
                                         self.simulation_directory,
                                         assign_principal_axes=True)

        self.first_time_in_Myr = self.first_snap.snapshot['time'] * 1000.0

        cluster_times = np.array([self.cluster[i]['time'] for i in
                                    range(len(self.cluster))])

        for i,idx in enumerate(self.snapshot_indices):


            if idx==self.start:
                self._ag_.update_index(idx, ss_id=self.ss_id,
                                       snap=self.first_snap)
            else:
                self._ag_.update_index(idx, ss_id=self.ss_id)

            current_time = self._ag_.snap.snapshot['time'] * 1000.0
            current_time -= self.first_time_in_Myr
            diff_time = np.abs(cluster_times - current_time)

            if np.min(diff_time) > 1.5 * self.timestep:
                # this means we've gone past the end of the run
                break

            cluster_key = np.argmin(diff_time)
            cl = self.cluster[cluster_key]

            actions = self._ag_.actions(cl['position'], cl['velocity'],
                                         add_ss=True)
            self.cluster[cluster_key]['actions'] = actions
            np.save('cluster_snapshots_actions.npy', self.cluster)

    def all_actions(self, fileout='cluster_snapshots_actions.p'):
        self._ag_.update_index(self.startnum, ss_id=self.ss_id)
        add_ss = not self.axisymmetric # if axi, we don't want to add ss
        for i,cl in enumerate(tqdm(self.cluster)):
            if not self.axisymmetric:
                self._ag_.update_ss(self.ss_id, position=cl['chosen_position']/1000.0,
                                    velocity=cl['chosen_velocity'])
            actions = self._ag_.actions(cl['position'], cl['velocity'],
                                        add_ss=add_ss)
            self.cluster[i]['actions'] = actions
        dill.dump(self.cluster, open(fileout, 'wb'))


class cluster_animator(object):
    def __init__(self, snapshots, xaxis='x', yaxis='y',
                 xmin=-0.01, xmax=0.01, ymin=-0.01, ymax=0.01,
                 start=None, end=None, fps=30, fileout=None,
                 mass_max=None, acc_map=False, interface=None, options=None,
                 nres=360, acc='tot', cmap='bwr_r', cmin=-0.5, cmax=0.5,
                 direction_arrow=False, plot_panel=False,
                 pLz_bound=2.0, pJr_bound=0.6, pJz_bound=0.1, normalize=False,
                 plot_cluster_com=False, com_rcut=None, color_by_dist=True,
                 dist_vmin = 0.0, dist_vmax=50.0, log_distance=False,
                 axisymmetric=False):

        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)
        mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


        self.snapshots = snapshots
        self.acc_map = acc_map
        self.acc = acc
        self.nres = nres
        self.cmap = cmap
        self.cmin = cmin
        self.cmax = cmax
        self.dist_vmin = dist_vmin
        self.dist_vmax = dist_vmax
        self.axisymmetric = axisymmetric

        self.direction_arrow = direction_arrow
        self.plot_panel = plot_panel
        self.plot_cluster_com = plot_cluster_com
        self.com_rcut = 0.8
        self.color_by_dist = color_by_dist
        self.log_distance = log_distance
        self._old_com_ = np.array([0, 0, 0])

        self.mass_max = mass_max

        self.xaxis = xaxis
        self.yaxis = yaxis
        self.normalize = normalize

        self._xaxis_key_ = self._axis_key_(self.xaxis)
        self._yaxis_key_ = self._axis_key_(self.yaxis)

        if start is None:
            self.start = 0
        else:
            self.start = start

        if end is None:
            self.end = len(self.snapshots)
        else:
            self.end = end

        self.fps = fps

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        first_x = self.snapshots[self.start]['position'][:, self._xaxis_key_]
        first_y = self.snapshots[self.start]['position'][:, self._yaxis_key_]
        first_mass = self.snapshots[self.start]['mass']

        if self.axisymmetric:
            first_x = np.subtract(first_x, np.median(first_x, axis=0))
            first_y = np.subtract(first_y, np.median(first_y, axis=0))

        if self.color_by_dist:
            pos = self.snapshots[self.start]['position']
            med_pos = np.median(pos, axis=0)
            diff = np.subtract(pos, med_pos)
            diff_mag = np.linalg.norm(diff, axis=1)
            if self.log_distance:
                c = np.log10(diff_mag)
            else:
                c = diff_mag
        else:
            c = 'k'

        if self.plot_panel:
            self.ax = plt.subplot(1, 2, 1)
            self.ax_traj = plt.subplot(2, 4, 3)
            self.ax_traj_vert = plt.subplot(2, 4, 4)
            self.ax_pJr = plt.subplot(2, 4, 7)
            self.ax_pJz = plt.subplot(2, 4, 8)

            self.fig = plt.gcf()
            self.fig.set_size_inches(10, 4)

            self.ax_pJr.set_xlim(-pLz_bound, pLz_bound)
            self.ax_pJr.set_ylim(-pJr_bound, pJr_bound)
            if self.normalize:
                self.ax_pJr.set_xlabel(r'$\text{p}L_z$')
                self.ax_pJr.set_ylabel(r'$\text{p}J_r$')
            else:
                self.ax_pJr.set_xlabel(r'$\Delta L_z\,[\text{kpc}\,\text{km/s}]$')
                self.ax_pJr.set_ylabel(r'$\Delta J_r\,[\text{kpc}\,\text{km/s}]$')

            self.ax_pJz.set_xlim(-pLz_bound, pLz_bound)
            self.ax_pJz.set_ylim(-pJz_bound, pJz_bound)
            if self.normalize:
                self.ax_pJz.set_xlabel(r'$\text{p}L_z$')
                self.ax_pJz.set_ylabel(r'$\text{p}J_z$')
            else:
                self.ax_pJz.set_xlabel(r'$\Delta L_z\,[\text{kpc}\,\text{km/s}]$')
                self.ax_pJz.set_ylabel(r'$\Delta J_z\,[\text{kpc}\,\text{km/s}]$')

            self.ax_traj.set_xlabel('x [kpc]')
            self.ax_traj.set_ylabel('y [kpc]')
            self.ax_traj_vert.set_xlabel('x [kpc]')
            self.ax_traj_vert.set_ylabel('z [kpc]')

            first_actions = self.snapshots[self.start]['actions']
            pecact = self._peculiar_actions_(first_actions)
            self.scat_pJr = self.ax_pJr.scatter(pecact[:,2], pecact[:,0],
                                                s=0.2, c=c, vmin=self.dist_vmin,
                                                vmax=self.dist_vmax)
            self.scat_pJz = self.ax_pJz.scatter(pecact[:,2], pecact[:,1],
                                                s=0.2, c=c, vmin=self.dist_vmin,
                                                vmax=self.dist_vmax)

            if self.axisymmetric:
                self.traj = \
                    np.array([np.median(self.snapshots[i]['position'], axis=0)
                              for i in range(len(self.snapshots))])
            else:
                self.traj = \
                    np.array([self.snapshots[i]['chosen_position'] for i in range(len(self.snapshots))])
            x = self.traj[:,0]
            y = self.traj[:,1]
            z = self.traj[:,2]
            self.ax_traj.plot(x, y, c='k', alpha=0.5)
            self.ax_traj_vert.plot(x, z, c='k', alpha=0.5)

            self.traj_scat = self.ax_traj.scatter(x[0], y[0], c='k', s=5)
            self.traj_vert_scat = self.ax_traj_vert.scatter(x[0], z[0], c='k', s=5)

            if self.start ==0:
                x = x[self.start]
                y = y[self.start]
                z = z[self.start]
            else:
                x = x[:self.start]
                y = y[:self.start]
                z = z[:self.start]
            self.traj_current, = self.ax_traj.plot(x, y, c='k')
            self.traj_vert_current, = self.ax_traj_vert.plot(x, z, c='k')

        else:
            self.fig, self.ax = plt.subplots(1)
        self.ax.axis('equal')
        self.scat = self.ax.scatter(first_x, first_y, s=first_mass, c=c,
                                    vmin=self.dist_vmin, vmax=self.dist_vmax)
        if self.color_by_dist:
            if self.log_distance:
                label = 'log10(dist) [pc]'
            else:
                label = 'dist [pc]'
            self.fig.colorbar(self.scat, ax=self.ax, label=label)

        if self.direction_arrow:
            chosen_velocity = self.snapshots[self.start]['chosen_velocity']
            vx = chosen_velocity[self._xaxis_key_]
            vy = chosen_velocity[self._yaxis_key_]
            init_mag = np.sqrt(vx*vx + vy*vy)
            self._arrow_norm_ = init_mag/(0.2 * self.xmax)
            self.arrow = self.ax.arrow(0, 0, vx/self._arrow_norm_,
                                       vy/self._arrow_norm_, fc="k", ec="k",
                                       head_width=0.05, head_length=0.1)

        if self.plot_cluster_com:
            pos = self.snapshots[self.start]['position']/1000.0 # convert to kpc
            mass = self.snapshots[self.start]['mass']
            com = self._cluster_com_(pos, mass)
            self.cluster_com_scat = self.ax.scatter(com[self._xaxis_key_], com[self._yaxis_key_], s=5, c='r')

        if acc_map:
            if interface is None or options is None:
                raise Exception('Please provide interface and options file')
                sys.exit(1)
            self.interface = interface
            self.gen_acc_map = acceleration_heatmap(options, interface)
            self.extent = [self.xmin, self.xmax, self.ymin, self.ymax]
            time = self.snapshots[self.start]['time']
            hm, hmx, hmy, hmz = self.gen_acc_map(time, index=self.start, return_heatmap=True,
                            plot_xmin=self.xmin, plot_xmax=self.xmax,
                            plot_ymin=self.ymin, plot_ymax=self.ymax,
                            nres=self.nres, cache=True)
            if self.acc == 'tot':
                this_hm = hm
            elif self.acc == 'x':
                this_hm = hmx
            elif self.acc == 'y':
                this_hm = hmy
            elif self.acc == 'z':
                this_hm = hmz

            self.im = self.ax.imshow(this_hm, extent=self.extent, origin='lower',
                            vmin=self.cmin, vmax=self.cmax, cmap=self.cmap,
                            animated=True)
            self.fig.colorbar(self.im, ax=self.ax)

        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)

        self.ax.set_xlabel(self.xaxis+' [kpc]')
        self.ax.set_ylabel(self.yaxis+' [kpc]')
        self.fig.tight_layout()

        if fileout is None:
            self.fileout = 'movie_' + self.xaxis + '_' + str(self.xmin)
            self.fileout += '_' + str(self.xmax) + '_' + self.yaxis + '_'
            self.fileout += str(self.ymin) + '_' + str(self.ymax)
            if self.mass_max is not None:
                self.fileout += '_massmax' + str(self.mass_max)
            self.fileout += '.mp4'
        else:
            self.fileout = fileout

    def _axis_key_(self, axis):
        if axis == 'x':
            return 0
        elif axis == 'y':
            return 1
        elif axis == 'z':
            return 2
        else:
            raise Exception('cant recognize axis: '+axis)
            sys.exit(1)

    def _animate_(self, frame, scat, im=None):
        this_x_data = self.snapshots[frame]['position'][:, self._xaxis_key_]/1000.0
        this_y_data = self.snapshots[frame]['position'][:, self._yaxis_key_]/1000.0
        this_mass = self.snapshots[frame]['mass']
        if self.axisymmetric:
            this_x_data = np.subtract(this_x_data, np.median(this_x_data))
            this_y_data = np.subtract(this_y_data, np.median(this_y_data))

        if self.color_by_dist:
            pos = self.snapshots[frame]['position']
            med_pos = np.median(pos, axis=0)
            diff = np.subtract(pos, med_pos)
            diff_mag = np.linalg.norm(diff, axis=1)
            if self.log_distance:
                c = np.log10(diff_mag)
            else:
                c = diff_mag

        if self.mass_max is not None:
            keys = np.where(this_mass < self.mass_max)[0]
            this_x_data = this_x_data[keys]
            this_y_data = this_y_data[keys]
            this_mass = this_mass[keys]

        if self.acc_map:
            time = self.snapshots[frame]['time']
            hm, hmx, hmy, hmz = self.gen_acc_map(time, index=frame, return_heatmap=True,
                            plot_xmin=self.xmin, plot_xmax=self.xmax,
                            plot_ymin=self.ymin, plot_ymax=self.ymax,
                            nres=self.nres, cache=True)
            if self.acc == 'tot':
                this_hm = hm
            elif self.acc == 'x':
                this_hm = hmx
            elif self.acc == 'y':
                this_hm = hmy
            elif self.acc == 'z':
                this_hm = hmz
            im.set_array(this_hm)

        if self.direction_arrow:
            vel = self.snapshots[frame]['chosen_velocity']
            vx = vel[self._xaxis_key_]
            vy = vel[self._yaxis_key_]
            vx /= self._arrow_norm_
            vy /= self._arrow_norm_
            self.arrow.set_xy(((0, 0), (vx, vy)))

        if self.plot_panel:
            this_actions = self.snapshots[frame]['actions']
            pact = self._peculiar_actions_(this_actions)
            self.scat_pJr.set_offsets(np.c_[pact[:,2], pact[:,0]])
            self.scat_pJz.set_offsets(np.c_[pact[:,2], pact[:,1]])
            if self.color_by_dist:
                self.scat_pJr.set_array(c)
                self.scat_pJz.set_array(c)

            x = self.traj[:,0]
            y = self.traj[:,1]
            z = self.traj[:,2]

            self.traj_scat.set_offsets(np.transpose([x[frame], y[frame]]))
            self.traj_vert_scat.set_offsets(np.transpose([x[frame], z[frame]]))

            if frame == 0:
                x = x[frame]
                y = y[frame]
                z = z[frame]
            else:
                x = x[:frame]
                y = y[:frame]
                z = z[:frame]
            self.traj_current.set_xdata(x)
            self.traj_current.set_ydata(y)
            self.traj_vert_current.set_xdata(x)
            self.traj_vert_current.set_ydata(z)

        # data = np.array([this_x_data, this_y_data])
        scat.set_offsets(np.c_[this_x_data, this_y_data])
        scat.set_sizes(this_mass)
        if self.color_by_dist:
            scat.set_array(c)
        t = self.snapshots[frame]['time']
        self.ax.set_title("{:.2f}".format(t))

        if self.plot_cluster_com:
            pos = self.snapshots[frame]['position']/1000.0 # convert to kpc
            mass = self.snapshots[frame]['mass']
            com = self._cluster_com_(pos, mass)
            self.cluster_com_scat.set_offsets(np.c_[com[self._xaxis_key_], com[self._yaxis_key_]])

        if self.acc_map:
            return (scat, im)
        else:
            return (scat,)

    def _peculiar_actions_(self, actions):
        med = np.median(actions, axis=0)
        p = np.subtract(actions, med)
        if self.normalize:
            p = np.divide(p, med)
        return p

    def _cluster_com_(self, position, mass):
        # this is weird - TODO: fix
        posmass = np.multiply(position, mass.reshape(-1, 1))
        totmass = np.sum(mass)
        if self.com_rcut is not None:
            for rcut in np.linspace(10*self.com_rcut, self.com_rcut, 100):
                diff = np.subtract(position, self._old_com_)
                diff_mag = np.linalg.norm(diff, axis=1)
                keys = np.where(diff_mag < rcut)[0]
                totmass = np.sum(mass[keys])
                com = np.sum(posmass[keys], axis=0)
                self._old_com_ = com/totmass

        self._old_com_ = com/totmass
        return self._old_com_

    def __call__(self):
        if self.acc_map:
            fargs = [self.scat, self.im]
        else:
            fargs = [self.scat]
        self.animation = FuncAnimation(self.fig, self._animate_,
                                       np.arange(self.start, self.end),
                                       fargs=fargs,
                                       interval=1000.0/self.fps,
                                       blit=False)
        self.animation.save(self.fileout, dpi=600)


class acceleration_heatmap(object):
    def __init__(self, options_file, interface):
        self.opt = options_reader(options_file)
        self.interface = interface

    def __call__(self, t_in_Myr, index=None, return_heatmap=False,
                 xcenter=0.0, ycenter=0.0, clim_min=-0.1, clim_max=0.1,
                 plot_xmin=-0.1, plot_xmax=0.1,
                 plot_ymin=-0.1, plot_ymax=0.1,
                 log=False, components=True, nres=360, zval=0.0,
                 output_file=None, cmap='YlGnBu', comp_cmap='bwr_r', cache=False):
                 # return_heatmap specifies to not plot but rather just
                 # return the heatmaps (used above)
                 # xcenter, ycenter are where the frame is centered
                 # clim_min, clim_max specify min and max values for
                 # color map
                 # plot_xmin, ... specify plot bounds
                 # log specifies whether to plot log(acc)
                 # note - will plot log(|acc_x|) for components
                 # components - return components in addition to tot acc
                 # output_file - if None, will make up a file name
                 # for components I will do 'x_' + output_file, etc...
                 # cmap - for tot acc, comp_cmap - for components
        if output_file is None:
            if log:
                output_file = 'logacc_'
            else:
                output_file = 'acc_'
            if index is not None:
                output_file += 'id'+str(index)+'_'
            else:
                output_file += 't'+str(t_in_Myr)+'_'
            output_file += 'xc' + str(xcenter) + '_yc' + str(ycenter)
            output_file += '_cmin' + str(clim_min) + '_cmax' + str(clim_max)
            output_file += '_plxmin' + str(plot_xmin) + '_plxmax' + str(plot_xmax)
            output_file += '_plymin' + str(plot_ymin) + '_plymax' + str(plot_ymax)
            output_file += '_nres' + str(nres) + '_zval' + str(zval) + '.pdf'

        self.components = components

        if components:
            output_file_x = 'x_' + output_file
            output_file_y = 'y_' + output_file
            output_file_z = 'z_' + output_file

        extent = [plot_xmin, plot_xmax, plot_ymin, plot_ymax]
        xlist = np.linspace(xcenter + plot_xmin, xcenter + plot_xmax, nres)
        ylist = np.linspace(ycenter + plot_ymin, ycenter + plot_ymax, nres)

        if components:
            heatmapx = np.zeros((nres, nres))
            heatmapy = np.zeros((nres, nres))
            heatmapz = np.zeros((nres, nres))
        heatmap = np.zeros((nres, nres))

        if cache:
            cache_directory = self.interface.cache_directory
            cache_file = cache_directory + '/' + output_file + '_cache.p'
            cache_file_x = cache_directory + '/' + output_file_x + '_cache.p'
            cache_file_y = cache_directory + '/' + output_file_y + '_cache.p'
            cache_file_z = cache_directory + '/' + output_file_z + '_cache.p'
            try:
                heatmap = dill.load(open(cache_file, 'rb'))
                if components:
                    heatmapx = dill.load(open(cache_file_x, 'rb'))
                    heatmapy = dill.load(open(cache_file_y, 'rb'))
                    heatmapz = dill.load(open(cache_file_z, 'rb'))
                print('found and loaded heatmap(s):', cache_file)

            except:
                print('couldnt find necessary heatmap(s) at cache:', cache_file)
                self.interface.evolve_model(t_in_Myr | units.Myr)
                heatmap, heatmapx, heatmapy, heatmapz = \
                    self._heatmap_(xlist, ylist, zval, heatmap, heatmapx,
                                   heatmapy, heatmapz)
                dill.dump(heatmap, open(cache_file, 'wb'))
                if components:
                    dill.dump(heatmapx, open(cache_file_x, 'wb'))
                    dill.dump(heatmapy, open(cache_file_y, 'wb'))
                    dill.dump(heatmapy, open(cache_file_z, 'wb'))
        else:
            self.interface.evolve_model(t_in_Myr | units.Myr)
            heatmap, heatmapx, heatmapy, heatmapz = \
                self._heatmap_(xlist, ylist, zval, heatmap, heatmapx,
                               heatmapy, heatmapz)

        if log:
            heatmapx = np.log10(np.abs(heatmapx))
            heatmapy = np.log10(np.abs(heatmapy))
            heatmapz = np.log10(np.abs(heatmapz))
            heatmap = np.log10(np.abs(heatmap))

        if return_heatmap:
            if components:
                return heatmap, heatmapx, heatmapy, heatmapz
            else:
                return heatmap

        self._plot_(heatmap, extent, t_in_Myr,
                    output_file, log, cmap, 0.0, clim_max)
        self._plot_(heatmapx, extent, t_in_Myr,
                    output_file_x, log, comp_cmap, clim_min, clim_max)
        self._plot_(heatmapy, extent, t_in_Myr,
                    output_file_y, log, comp_cmap, clim_min, clim_max)
        self._plot_(heatmapz, extent, t_in_Myr,
                    output_file_z, log, comp_cmap, clim_min, clim_max)

    def _heatmap_(self, xlist, ylist, zval, heatmap, heatmapx, heatmapy, heatmapz):
        for i,x in enumerate(tqdm(xlist)):
            print('got to:', i)
            for j,y in enumerate(ylist):
                acc = self.interface.get_gravity_at_point(0 | units.kpc,
                x | units.kpc, y | units.kpc, zval | units.kpc)
                acc = np.array([acc[i].value_in(units.kms/units.Myr) for i in range(3)])
                if self.components:
                    heatmapx[j][i] = acc[0]
                    heatmapy[j][i] = acc[1]
                    heatmapz[j][i] = acc[2]
                heatmap[j][i] = np.linalg.norm(acc)
        return heatmap, heatmapx, heatmapy, heatmapz

    def _plot_(self, heatmap, extent, t, out, log, cmap, clim_min, clim_max):
        plt.imshow(heatmap, extent=extent, origin='lower', cmap=cmap,
                   vmin=clim_min, vmax=clim_max)
        if log:
            plt.colorbar(label='log10(acc) [ km/s/Myr ]')
        else:
            plt.colorbar(label='acc [ km/s/Myr ]')
        plt.xlabel('x [kpc]')
        plt.ylabel('y [kpc]')
        plt.xlim(extent[0], extent[1])
        plt.ylim(extent[2], extent[3])
        plt.title("{:10.2f}".format(t) + ' Myr')
        plt.savefig(out)
        plt.close()


"""
if __name__ == '__main__':
    from oceanic.options import options_reader
    opt = options_reader(sys.argv[1])
    ag = agama_wrapper(opt)
"""
"""
if __name__ == '__main__':
    from oceanic.options import options_reader
    opt = options_reader(sys.argv[1])
    ss_id = int(sys.argv[2])
    cl_act = snapshot_action_calculator(opt, ss_id=ss_id)
    cl_act.scroll_actions()
"""
