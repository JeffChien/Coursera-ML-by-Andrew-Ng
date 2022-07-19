#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this fixes 2 things
# first it turns off 'bracketed paste mode' no more [?2004l
# second, much better performance
# http://savannah.gnu.org/bugs/?59483#comment5
c.OctaveKernel.cli_options = ' --no-line-editing'
