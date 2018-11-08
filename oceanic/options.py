from configparser import ConfigParser


class options_reader(object):
    def __init__(self, file):
        self.parser = ConfigParser(inline_comment_prefixes=';')
        try:
            self.parser.read(file)
        except:
            print('couldnt read options file')
            raise Exception('Cant read the provided options file:' + file)

        self.options = {}

        # read in general parameters
        for opt in ['output_directory']:
            self._read_required_option_('general', opt)

        self._read_optional_option_('general', 'ncpu', '1')
        self._read_optional_option_('general', 'gpu_enabled', 'false')
        self._read_optional_option_('general', 'ngpu', '1')
        self._read_optional_option_('general', 'write_frequency', '10')

        # read in simulation parameters
        for opt in ['simulation_directory', 'cache_directory', 'startnum',
                    'endnum',
                    'star_softening_in_pc', 'dark_softening_in_pc']:
            self._read_required_option_('simulation', opt)

        self._read_optional_option_('simulation', 'num_prior', '3')
        self._read_optional_option_('simulation', 'sim_name', None)
        self._read_optional_option_('simulation', 'star_char_mass', None)
        self._read_optional_option_('simulation', 'dark_char_mass', None)
        if self.options['star_char_mass'] is not None:
            self.options['star_char_mass'] = float(self.options['star_char_mass'])
        if self.options['dark_char_mass'] is not None:
            self.options['dark_char_mass'] = float(self.options['dark_char_mass'])

        # read in force_calculation parameters
        for opt in ['Rmax', 'theta']:
            self._read_required_option_('force_calculation', opt)

        self._read_optional_option_('force_calculation', 'nclose', '150')
        self._read_optional_option_('force_calculation', 'basis', 'phs3')
        self._read_optional_option_('force_calculation', 'order', '5')
        self._read_optional_option_('force_calculation', 'eps', '1.0')

        self._read_optional_option_('force_calculation', 'axisymmetric', 'false')
        self.options['axisymmetric'] = self._convert_bool_(self.options['axisymmetric'])
        if self.options['axisymmetric']:
            self._read_required_option_('force_calculation', 'axi_Rinit')
            self._read_optional_option_('force_calculation', 'axi_vcircfrac', 1.0)
            self._read_optional_option_('force_calculation', 'axi_zinit', 0)
            self.options['axi_Rinit'] = float(self.options['axi_Rinit'])
            self.options['axi_vcircfrac'] = float(self.options['axi_vcircfrac'])
            self.options['axi_zinit'] = float(self.options['axi_zinit'])

        # read in cluster parameters
        for opt in ['N', 'W0', 'Rcluster', 'softening',
                    'nbodycode', 'use_kroupa',
                    'timestep', 'tend']:
            self._read_required_option_('cluster', opt)

        self._read_optional_option_('cluster', 'eject_cut', '300.0')
        self._read_optional_option_('cluster', 'Mcluster', None)
        self._read_optional_option_('cluster', 'kroupa_max', '100.0')

        # read in starting_star parameters
        # for opt in ['ss_Rmin', 'ss_Rmax', 'ss_zmin', 'ss_zmax']:
        #    self._read_required_option_('starting_star', opt)

        self._read_optional_option_('starting_star', 'ss_Rmin', None)
        self._read_optional_option_('starting_star', 'ss_Rmax', None)
        self._read_optional_option_('starting_star', 'ss_zmin', None)
        self._read_optional_option_('starting_star', 'ss_zmax', None)

        self._read_optional_option_('starting_star', 'ss_agemin_in_Gyr', '0')
        self._read_optional_option_('starting_star', 'ss_agemax_in_Gyr', '0')
        self._read_optional_option_('starting_star', 'ss_seed', '1776')
        self._read_optional_option_('starting_star', 'ss_id', None)

        self._read_optional_option_('starting_star', 'ss_action_cuts', 'false')
        self.options['ss_action_cuts'] =\
            self._convert_bool_(self.options['ss_action_cuts'])
        if self.options['ss_action_cuts']:
            for opt in ['Jr_min', 'Jr_max', 'Jz_min', 'Jz_max']:
                self._read_required_option_('starting_star', opt)

        ss_array = [self.options['ss_Rmin'], self.options['ss_Rmax'],
                    self.options['ss_zmin'], self.options['ss_zmax']]

        if self.options['ss_id'] is None and None in ss_array:
            print('if ss_id is not given, you must provide Rmin, Rmax,')
            print('zmin, zmax. Exiting...')
            raise Exception('insufficient information to determine ss')
        elif self.options['ss_id'] is not None:
            self.options['ss_id'] = int(self.options['ss_id'])

        # read in grid parameters
        for opt in ['grid_x_size_in_kpc', 'grid_y_size_in_kpc',
                    'grid_z_size_in_kpc', 'grid_resolution']:
            self._read_required_option_('grid', opt)

        self._read_optional_option_('grid', 'grid_seed', '1776')
        self._read_optional_option_('grid', 'grid_fine_x_size_in_kpc', None)
        self._read_optional_option_('grid', 'grid_fine_y_size_in_kpc', None)
        self._read_optional_option_('grid', 'grid_fine_z_size_in_kpc', None)
        self._read_optional_option_('grid', 'grid_fine_resolution', None)

        if self.options['grid_fine_x_size_in_kpc'] is not None:
            self.options['fine_grid'] = True
            self.options['grid_fine_x_size_in_kpc'] = float(self.options['grid_fine_x_size_in_kpc'])
            self.options['grid_fine_y_size_in_kpc'] = float(self.options['grid_fine_y_size_in_kpc'])
            self.options['grid_fine_z_size_in_kpc'] = float(self.options['grid_fine_z_size_in_kpc'])
            self.options['grid_fine_resolution'] = float(self.options['grid_fine_resolution'])
        else:
            self.options['fine_grid'] = False

        # convert relevant parameters
        int_options = ['startnum', 'endnum', 'num_prior', 'nclose', 'order',
                       'ss_seed', 'write_frequency',
                       'ncpu', 'ngpu', 'N', 'W0', 'grid_seed']
        for opt in int_options:
            if opt in self.options.keys():
                self.options[opt] = int(self.options[opt])

        float_options = ['ss_Rmin', 'ss_Rmax', 'ss_zmin', 'ss_zmax',
                         'ss_agemin_in_Gyr', 'ss_agemax_in_Gyr', 'Mcluster',
                         'Rcluster',
                         'softening', 'eject_cut', 'timestep', 'tend', 'eps'
                         'star_softening_in_pc', 'dark_softening_in_pc',
                         'Rmax', 'theta',
                         'grid_x_size_in_kpc', 'grid_y_size_in_kpc',
                         'grid_z_size_in_kpc', 'grid_resolution', 'kroupa_max',
                         'Jr_min', 'Jr_max', 'Jz_min', 'Jz_max']
        for opt in float_options:
            if opt in self.options.keys():
                self.options[opt] = float(self.options[opt])

        bool_options = ['gpu_enabled', 'use_kroupa']
        for opt in bool_options:
            if opt in self.options.keys():
                self.options[opt] = self._convert_bool_(self.options[opt])

        self._convert_rbf_basis_(self.options['basis'])
        self._convert_nbodycode_(self.options['nbodycode'])

    def set_options(self, object):
        for key in self.options.keys():
            setattr(object, key, self.options[key])

    def _convert_bool_(self, string):
        if string in ['True', 'true', '1', 1]:
            return True
        elif string in ['False', 'false', '0', 0]:
            return False
        else:
            raise Exception("Can't recognize bool:", string, "in options file")

    def _read_required_option_(self, category, option):
        try:
            self.options[option] = self.parser.get(category, option)
            print('set option: ', option, ' as: ', self.options[option])
        except:
            raise Exception('Couldnt find required option: ' + option)

    def _read_optional_option_(self, category, option, default):
        try:
            self.options[option] = self.parser.get(category, option)
            print('set option: ', option, ' as: ', self.options[option])
        except:
            print('Couldnt find option', option, ', using default: ', default)
            self.options[option] = default

    def _convert_rbf_basis_(self, basis_string):
        if basis_string == 'phs8':
            from rbf.basis import phs8
            self.options['basis'] = phs8
        elif basis_string == 'phs7':
            from rbf.basis import phs7
            self.options['basis'] = phs7
        elif basis_string == 'phs6':
            from rbf.basis import phs6
            self.options['basis'] = phs6
        elif basis_string == 'phs5':
            from rbf.basis import phs5
            self.options['basis'] = phs5
        elif basis_string == 'phs4':
            from rbf.basis import phs4
            self.options['basis'] = phs4
        elif basis_string == 'phs3':
            from rbf.basis import phs3
            self.options['basis'] = phs3
        elif basis_string == 'phs2':
            from rbf.basis import phs2
            self.options['basis'] = phs2
        elif basis_string == 'phs1':
            from rbf.basis import phs1
            self.options['basis'] = phs1
        elif basis_string == 'mq':
            from rbf.basis import mq
            self.options['basis'] = mq
        elif basis_string == 'imq':
            from rbf.basis import imq
            self.options['basis'] = imq
        elif basis_string == 'iq':
            from rbf.basis import iq
            self.options['basis'] = iq
        elif basis_string == 'ga':
            from rbf.basis import ga
            self.options['basis'] = ga
        elif basis_string == 'exp':
            from rbf.basis import exp
            self.options['basis'] = exp
        elif basis_string == 'se':
            from rbf.basis import se
            self.options['basis'] = se
        elif basis_string == 'mat32':
            from rbf.basis import mat32
            self.options['basis'] = mat32
        elif basis_string == 'mat52':
            from rbf.basis import mat52
            self.options['basis'] = mat52
        elif basis_string == 'wen10':
            from rbf.basis import wen10
            self.options['basis'] = wen10
        elif basis_string == 'wen11':
            from rbf.basis import wen11
            self.options['basis'] = wen11
        elif basis_string == 'wen12':
            from rbf.basis import wen12
            self.options['basis'] = wen12
        elif basis_string == 'wen30':
            from rbf.basis import wen30
            self.options['basis'] = wen30
        elif basis_string == 'wen31':
            from rbf.basis import wen31
            self.options['basis'] = wen31
        elif basis_string == 'wen32':
            from rbf.basis import wen32
            self.options['basis'] = wen32
        else:
            raise Exception("Can't recognize given basis: "+basis_string)

    def _convert_nbodycode_(self, code_string):
        if code_string == 'ph4':
            from amuse.community.ph4.interface import ph4
            self.options['nbodycode'] = ph4
        else:
            raise Exception("Can't recognize given code: "+code_string)


if __name__ == '__main__':
    import sys
    g = options_reader(sys.argv[1])
