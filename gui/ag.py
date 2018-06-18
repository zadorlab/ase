# Copyright 2008, 2009
# CAMd (see accompanying license files for details).
from __future__ import print_function, unicode_literals


class CLICommand:
    short_description = "ASE's graphical user interface"
    description = ('ASE-GUI.  See the online manual '
                   '(https://wiki.fysik.dtu.dk/ase/ase/gui/gui.html) '
                   'for more information.')

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('filenames', nargs='*')
        add('-n', '--image-number',
            default=':', metavar='NUMBER',
            help='Pick image(s) from trajectory.  NUMBER can be a '
            'single number (use a negative number to count from '
            'the back) or a range: start:stop:step, where the '
            '":step" part can be left out - default values are '
            '0:nimages:1.')
        add('-u', '--show-unit-cell', type=int,
            default=1, metavar='I',
            help="0: Don't show unit cell.  1: Show unit cell.  "
            '2: Show all of unit cell.')
        add('-r', '--repeat',
            default='1',
            help='Repeat unit cell.  Use "-r 2" or "-r 2,3,1".')
        add('-R', '--rotations', default='',
            help='Examples: "-R -90x", "-R 90z,-30x".')
        add('-o', '--output', metavar='FILE',
            help='Write configurations to FILE.')
        add('-g', '--graph',
            # TRANSLATORS: EXPR abbreviates 'expression'
            metavar='EXPR',
            help='Plot x,y1,y2,... graph from configurations or '
            'write data to sdtout in terminal mode.  Use the '
            'symbols: i, s, d, fmax, e, ekin, A, R, E and F.  See '
            'https://wiki.fysik.dtu.dk/ase/ase/gui/gui.html'
            '#plotting-data for more details.')
        add('-t', '--terminal',
            action='store_true',
            default=False,
            help='Run in terminal window - no GUI.')
        add('--interpolate',
            type=int, metavar='N',
            help='Interpolate N images between 2 given images.')
        add('-b', '--bonds',
            action='store_true',
            default=False,
            help='Draw bonds between atoms.')
        add('-s', '--scale', dest='radii_scale', metavar='FLOAT',
            default=None, type=float,
            help='Scale covalent radii.')

    @staticmethod
    def run(args):
        from ase.gui.images import Images
        from ase.atoms import Atoms

        images = Images()

        if args.filenames:
            from ase.io import string2index
            images.read(args.filenames, string2index(args.image_number))
        else:
            images.initialize([Atoms()])

        if args.interpolate:
            images.interpolate(args.interpolate)

        if args.repeat != '1':
            r = args.repeat.split(',')
            if len(r) == 1:
                r = 3 * r
            images.repeat_images([int(c) for c in r])

        if args.radii_scale:
            images.set_radii(args.radii_scale)

        if args.output is not None:
            images.write(args.output, rotations=args.rotations,
                         show_unit_cell=args.show_unit_cell)
            args.terminal = True

        if args.terminal:
            if args.graph is not None:
                data = images.graph(args.graph)
                for line in data.T:
                    for x in line:
                        print(x, end=' ')
                    print()
        else:
            from ase.gui.gui import GUI
            gui = GUI(images, args.rotations, args.show_unit_cell, args.bonds)
            gui.run(args.graph)