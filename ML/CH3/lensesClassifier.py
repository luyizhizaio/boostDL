# -*- coding: utf-8 -*-
__author__ = 'tend'


import trees

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = trees.createTree(lenses,lensesLabels)

#execute
print lensesTree

import treePlotter

treePlotter.createPlot(lensesTree)














