import unittest
import math
import pickle
import os

import numpy
from numpy import testing
import numericalunits

from dfttools.types import *

class BasisInitializationTest(unittest.TestCase):
    
    def test_init_0(self):
        
        b = Basis(
            ((1,0),(0,1))
        )
        testing.assert_equal(b.vectors, numpy.array(
            ((1,0),(0,1))
        ))
        
    def test_init_1(self):
        b = Basis(
            numpy.array(((1,2,3),(4,5,6),(7,8,9)))
        )
        testing.assert_equal(b.vectors, numpy.array(
            ((1,2,3),(4,5,6),(7,8,9))
        ))
        
    def test_init_2(self):
        b = Basis(
            (1,2,3),
            kind = 'orthorombic'
        )
        testing.assert_equal(b.vectors, numpy.array(
            ((1,0,0),(0,2,0),(0,0,3))
        ))
        
    def test_init_3(self):
        b = Basis(
            (1,1,3,0,0,0.5),
            kind = 'triclinic'
        )
        testing.assert_allclose(b.vectors, numpy.array(
            ((1,0,0),(0.5,3.**.5/2,0),(0,0,3))
        ), atol = 1e-7)
        
    def test_init_4(self):
        a = numpy.array(((1,0),(0,1)))
        meta = {"a":"b"}
        b = Basis(a, meta = meta)
        assert not b.vectors is a
        assert not b.meta is meta
        testing.assert_equal(b.meta, meta)
        
    def test_init_raises(self):
        with self.assertRaises(ArgumentError):
            Basis(((1,0),(0,1)), kind = 'unknown')
            
class BasisTest(unittest.TestCase):
    
    def setUp(self):
        self.b = Basis(
            (1,1,3,0,0,0.5),
            kind = 'triclinic',
            meta = {"key":"value"},
        )
        self.c = Basis(
            (1,1,1),
            kind = 'orthorombic',
        )
        
    def test_pickle(self):
        b = pickle.loads(pickle.dumps(self.b))
        testing.assert_equal(b.vectors, self.b.vectors)
        testing.assert_equal(b.meta, self.b.meta)
        
    def test_eq(self):
        assert self.b == Basis(self.b.vectors)
        assert not self.b == self.c
        
    def test_transform(self):
        coordinates = numpy.array((1,1,1))
        
        transformed = self.b.transform_to(self.c, coordinates)
        testing.assert_allclose(transformed, (1.5,3.**.5/2,3), atol = 1e-7)
        
        transformed = self.b.transform_from(self.c, transformed)
        testing.assert_allclose(transformed, coordinates, atol = 1e-7)
        
        transformed = self.b.transform_to_cartesian(coordinates)
        testing.assert_allclose(transformed, (1.5,3.**.5/2,3), atol = 1e-7)
        
        transformed = self.b.transform_from_cartesian(transformed)
        testing.assert_allclose(transformed, coordinates, atol = 1e-7)
        
        transformed = self.b.transform_to(self.c, coordinates[numpy.newaxis,:])
        testing.assert_allclose(transformed, ((1.5,3.**.5/2,3),), atol = 1e-7)
        
    def test_volume(self):
        assert abs(self.b.volume() - 1.5*3.**.5) < 1e-7
    
    def test_reciprocal(self):
        testing.assert_allclose(numpy.dot(numpy.swapaxes(self.b.reciprocal().vectors,0,1),self.b.vectors),numpy.eye(3), atol = 1e-10)
        
    def test_vertices(self):
        v = self.b.vertices()
        s = 0.5*3.**.5
        testing.assert_allclose(v, (
            (0,0,0),
            (0,0,3),
            (.5, s, 0),
            (.5, s, 3),
            (1,0,0),
            (1,0,3),
            (1.5, s, 0),
            (1.5, s, 3),
        ), atol = 1e-7)

    def test_edges(self):
        v = self.b.edges()
        s = 0.5*3.**.5
        testing.assert_allclose(v, (
            ((0,0,0),(1,0,0)),
            ((0,0,3),(1,0,3)),
            ((.5, s, 0),(1.5, s, 0)),
            ((.5, s, 3),(1.5, s, 3)),
            ((0,0,0),(.5,s,0)),
            ((0,0,3),(.5,s,3)),
            ((1,0,0),(1.5,s,0)),
            ((1,0,3),(1.5,s,3)),
            ((0, 0, 0),(0, 0, 3)),
            ((.5, s, 0),(.5, s, 3)),
            ((1, 0, 0),(1, 0, 3)),
            ((1.5, s, 0),(1.5, s, 3)),
        ), atol = 1e-7)
        
    def test_copy(self):
        b2 = self.b.copy()
        testing.assert_equal(self.b.vectors, b2.vectors)
        testing.assert_equal(self.b.meta, b2.meta)
        assert not self.b.vectors is b2.vectors
        assert not self.b.meta is b2.meta
        
    def test_stack(self):
        nv = self.b.vectors.copy()
        
        st = self.b.stack(self.b, vector = 'y')
        nv[1] *= 2
        testing.assert_allclose(st.vectors,nv)
        
        st = st.stack(st, st, vector = 'x')
        nv[0] *= 3
        testing.assert_allclose(st.vectors,nv)
        
    def test_stack_raises(self):
        with self.assertRaises(ArgumentError):
            self.b.stack(self.c, vector = 0)
            
    def test_repeated(self):
        nv = self.b.vectors.copy()
        
        r = self.b.repeated(3,2,1)
        nv[0] *= 3
        nv[1] *= 2
        testing.assert_allclose(r.vectors,nv)
        
    def test_repeated_fail(self):
        with self.assertRaises(ValueError):
            rp = self.b.repeated(2.0,2.0,1.0)
        
    def test_reorder(self):
        c = self.b.copy()
        
        c.reorder_vectors(0,1,2)
        testing.assert_equal(self.b.vectors, c.vectors)
        
        c.reorder_vectors(2,1,0)
        testing.assert_equal(self.b.vectors[::-1], c.vectors)
        
        with self.assertRaises(ArgumentError):
            c.reorder_vectors(2,1)
            
        with self.assertRaises(ArgumentError):
            c.reorder_vectors(2,1,0,3)
        
        with self.assertRaises(ArgumentError):
            c.reorder_vectors(2,1,2)
            
    def test_genpath(self):
        keys = ((0,0,0),(1,0,0),(1,1,0),(1,1,1))
        pth = self.c.generate_path(keys, 7)
        pth2 = self.c.generate_path(keys, 7, anchor = False)
        testing.assert_allclose(pth, pth2)
        testing.assert_allclose(pth, (
            (0,0,0),
            (.5,0,0),
            (1,0,0),
            (1,.5,0),
            (1,1,0),
            (1,1,.5),
            (1,1,1)
        ))
        
    def test_genpath2(self):
        keys = ((0,0,0),(1,0,0),(1,1,0),(1,1,1))
        pth = self.c.generate_path(keys, 7, anchor = True)
        for k in keys:
            a = numpy.array(k, dtype = float).tolist()
            b = pth.tolist()
            assert a in b
        
    def test_save_load(self):
        import numericalunits
        import pickle
        x = old = Basis(
            (numericalunits.angstrom,)*3,
            kind = 'orthorombic',
            units = 'angstrom',
        )
        assert x.units_aware()
        data = pickle.dumps(x)
        numericalunits.reset_units()
        x = pickle.loads(data)
        # Assert object changed
        assert x != old
        testing.assert_allclose(x.vectors, numpy.eye(3)*numericalunits.angstrom)
        
    def test_save_load_json(self):
        import json
        x = old = Basis(
            (numericalunits.angstrom,)*3,
            kind = 'orthorombic',
            units = 'angstrom',
        )
        assert x.units_aware()
        data = json.dumps(x.to_json())
        numericalunits.reset_units()
        x = Basis.from_json(json.loads(data))
        # Assert object changed
        assert x != old
        testing.assert_allclose(x.vectors, numpy.eye(3)*numericalunits.angstrom)
    
    def test_rotated(self):
        b1 = self.b.rotated((0,0,-1),numpy.pi/2)
        testing.assert_allclose(b1.vectors, (
            (0,1,0),
            (-3.**.5/2,.5,0),
            (0,0,3),
        ), atol = 1e-7)
        
    def test_rotated_2(self):
        c1 = self.c.rotated((0,0,-1),numpy.pi/4)
        s2 = 1./2.**.5
        testing.assert_allclose(c1.vectors, (
            (s2,s2,0),
            (-s2,s2,0),
            (0,0,1),
        ))
        
class BasisMacroTests(unittest.TestCase):
    
    def test_diamond(self):
        a = diamond_basis(3.14)
        testing.assert_allclose(a.volume(), 3.14**3/4)

class CellInitializationTest(unittest.TestCase):
    
    def setUp(self):
        self.a = 2.510e-10
        self.h = self.a*(2./3.)**0.5
        self.hcpCo_b = Basis((self.a,self.a,self.a,0.5,0.5,0.5), kind = 'triclinic')
    
    def test_init_0(self):
        
        hcpCo = UnitCell(self.hcpCo_b, (0,0,0), 'Co')
        assert hcpCo.values.shape == (1,)
        assert hcpCo.coordinates.shape == (1,3)
        assert numpy.all(hcpCo.values == "Co")
        assert numpy.all(hcpCo.coordinates == 0)
        testing.assert_allclose((hcpCo.vectors**2).sum(axis = 1),self.a**2)
        
    def test_init_1(self):
        
        hcpCo = UnitCell(
            Basis((self.a,self.a,self.h*2,0.,0.,0.5), kind = 'triclinic'),
            ((0.,0.,0.),(1./3.,1./3.,0.5)),
            'Co',
        )
        assert hcpCo.values.shape == (2,)
        assert hcpCo.coordinates.shape == (2,3)
        assert numpy.all(hcpCo.values == "Co")
        testing.assert_allclose(hcpCo.coordinates,((0.,0.,0.),(1./3.,1./3.,0.5)))
        
    def test_init_2(self):
        
        n = UnitCell(
            Basis((1e-10,1e-10,1e-10), kind = 'orthorombic'),
            ((.25,.5,.5),(.5,.25,.5),(.5,.5,.25),(.25,.25,.25)),
            ((1,),(2,))
        )
        assert n.values.shape == (4,1)
        assert n.values[3,0] == 2
        
    def test_init_3(self):
        
        n = UnitCell(
            Basis((1e-10,1e-10,1e-10), kind = 'orthorombic'),
            (.5,.5,.5),
            'N',
            c_basis=Basis(((2e-10,0,0),(0,1e-10,0),(0,0,3e-10)))
        )
        testing.assert_equal(n.coordinates,((1.,.5,1.5),))
        
    def test_init_4(self):
        
        n = UnitCell(
            Basis((1,2,3), kind = 'orthorombic'),
            (.5,.5,.5),
            'N',
            c_basis='cartesian'
        )
        testing.assert_equal(n.coordinates,((.5,.25,1./6),))
        
    def test_init_fail_broadcast(self):
        with self.assertRaises(ArgumentError):
            UnitCell(
                Basis((1,1,1), kind = 'orthorombic'),
                ((.25,.5,.5),(.5,.25,.5),(.5,.5,.25),(.25,.25,.25)),
                (1,2,3)
            )
        
    def test_init_fail_size_0(self):
        with self.assertRaises(ArgumentError):
            UnitCell(
                Basis((1,1,1), kind = 'orthorombic'),
                (.25,.5,.5,.5),
                'C',
            )
            
    def test_init_fail_size_1(self):
        with self.assertRaises(ArgumentError):
            UnitCell(
                Basis((1,1,1), kind = 'orthorombic'),
                (
                    (.25,.5,.5,.5),
                    (.25,.5,.5,.5),
                ),
                'C',
            )

class CellTest(unittest.TestCase):
    
    @staticmethod
    def __co__(a,h,**kwargs):
        return UnitCell(
            Basis((a, a, h, 0., 0., 0.5), kind = 'triclinic'),
            ((0.,0.,0.),(1./3.,1./3.,0.5)),
            'Co',
            **kwargs
        )
        
    def setUp(self):
        self.a = 2.510e-10
        self.h = 2*self.a*(2./3.)**0.5
        self.cell = CellTest.__co__(self.a,self.h)
        self.cell2 = UnitCell(
            Basis((self.a,self.a,self.a,0.5,0.5,0.5), kind = 'triclinic'),
            (0,0,0),
            'Co'
        )
        self.empty = Basis((self.a,self.a,self.a,0.5,0.5,0.5), kind = 'triclinic')
        
    def test_eq(self):
        c = self.cell.copy()
        assert self.cell == c
        c.coordinates[0,0] = 3.14
        assert not self.cell == c
        c.coordinates[0,0] = 0
        assert self.cell == c
        c.values[0] = 'x'
        assert not self.cell == c
        
    def test_volume_0(self):
        assert abs(self.cell.volume()/(.5*3.**.5*self.a**2*self.h)-1)<1e-7
        
    def test_volume_1(self):
        assert abs(self.cell.volume()/(2*self.cell2.volume())-1)<1e-7
        
    def test_size(self):
        assert self.cell.size() == 2*self.cell2.size() == 2
        
    def test_copy_0(self):
        cp = self.cell.copy()
        assert not (cp.vectors is self.cell.vectors)
        assert not (cp.coordinates is self.cell.coordinates)
        assert not (cp.values is self.cell.values)
        testing.assert_equal(cp.vectors,self.cell.vectors)
        testing.assert_equal(cp.coordinates,self.cell.coordinates)
        testing.assert_equal(cp.values,self.cell.values)
        
    def test_cut_0(self):
        cp = self.cell.cut(0.,0.,0.,1.,1.,1.)
        testing.assert_allclose(cp.vectors,self.cell.vectors)
        testing.assert_allclose(cp.coordinates,self.cell.coordinates)
        testing.assert_equal(cp.values,self.cell.values)
        
    def test_cut_1(self):
        cp = self.cell.cut(0.,0.,0.,.5,.5,.5)
        testing.assert_allclose((cp.vectors**2).sum(axis=1)*4,(self.a**2,self.a**2,self.h**2))
        testing.assert_equal(cp.coordinates,((0.,0.,0.),))
        testing.assert_equal(cp.values,("Co",))
        
    def test_cut_2(self):
        cp = self.cell.cut(.25,.25,.25,.75,.75,.75)
        testing.assert_allclose((cp.vectors**2).sum(axis=1)*4,(self.a**2,self.a**2,self.h**2))
        testing.assert_allclose(cp.coordinates,((1./6,1./6,1./2),))
        testing.assert_equal(cp.values,("Co",))
        
    def test_cut_3(self):
        cp = self.cell.cut(.25,.75,.75,.75,.25,.25)
        testing.assert_allclose((cp.vectors**2).sum(axis=1)*4,(self.a**2,self.a**2,self.h**2))
        testing.assert_allclose(cp.coordinates,((1./6,1./6,1./2),))
        testing.assert_equal(cp.values,("Co",))
        
    def test_cartesian(self):
        testing.assert_allclose(self.cell.cartesian(),((0.,0.,0.), (self.a/2,self.a/2/3.**.5,self.h/2)))
            
    def test_angles(self):
        supercell = self.cell.repeated(2,2,1)
        testing.assert_allclose(supercell.angles(0,2,4,6),(.5,)*2)
        testing.assert_allclose(supercell.angles(1,3,5,7),(.5,)*2)
        testing.assert_allclose(supercell.angles((0,4,6),(0,2,6)),(-.5,)*2)
        testing.assert_allclose(supercell.angles(numpy.array(((0,4,6),(0,2,6)))),(-.5,)*2)
        with self.assertRaises(ArgumentError):
            self.cell.angles( ((0,1,2),(1,2,3)), ((2,3,4),(2,4,0)) )
        with self.assertRaises(ArgumentError):
            self.cell.angles((0,1,2,3),(1,2,3,4))
        with self.assertRaises(ArgumentError):
            self.cell.angles(0,1)
        with self.assertRaises(ArgumentError):
            self.cell.angles(1,1,1,1)
            
    def test_distances_0(self):
        d = (self.a**2/3+self.h**2/4)**.5
        testing.assert_allclose(self.cell.distances(0,1,0,1),(d,)*3)
        with self.assertRaises(ArgumentError):
            self.cell.distances(0)
        testing.assert_allclose(self.cell.distances(),(
            (0,d),
            (d,0)
        ))
    
    def test_distances_1(self):
        supercell = self.cell.repeated(2,2,1)
        testing.assert_allclose(supercell.distances(0,2,4,6),(self.a,)*3)
        testing.assert_allclose(supercell.distances(1,3,5,7),(self.a,)*3)
        testing.assert_allclose(supercell.distances((0,4),(2,6)),(self.a,)*2)
        testing.assert_allclose(supercell.distances(numpy.array(((0,6),))),(self.a*3**.5,))
        with self.assertRaises(ArgumentError):
            self.cell.distances((0,))
        with self.assertRaises(ArgumentError):
            self.cell.distances((0,1,2),(1,2,3))
        with self.assertRaises(ArgumentError):
            self.cell.distances((0,1,2,3), (1,2,3,4))
        with self.assertRaises(ArgumentError):
            self.cell.distances(((0,1,2),(1,2,3)), ((2,3,4),(3,4,0)))
            
    def test_isolated2_0(self):
        gap = 1e-10
        iso = self.cell2.isolated2(gap)
        testing.assert_allclose(iso.distances(),self.cell2.distances())
        
    def test_isolated2_1(self):
        gap = 1e-10
        iso = self.cell.isolated2(gap)
        testing.assert_allclose(iso.distances(),self.cell.distances())
        
    def test_stack_0(self):
        st = self.cell.stack(self.cell, vector = 'z')
        nv = self.cell.vectors.copy()
        nv[2,:] *= 2
        testing.assert_allclose(st.vectors,nv)
        testing.assert_allclose(st.coordinates,
            ((0.,0.,0.),(1./3.,1./3.,0.25),(0.,0.,.5),(1./3.,1./3.,0.75)))
        testing.assert_equal(st.values,('Co',)*4)
        
    def test_stack_1(self):
        st = self.cell2.stack(self.cell2, vector = 0)
        nv = self.cell2.vectors.copy()
        nv[0,:] *= 2
        testing.assert_allclose(st.vectors,nv)
        testing.assert_allclose(st.coordinates,
            ((0.,0.,0.),(.5,0.,0.)))
        testing.assert_equal(st.values,('Co',)*2)

    def test_stack_2(self):
        st = self.cell2.stack(self.empty, self.cell2)
        nv = self.cell2.vectors.copy()
        nv[0,:] *= 3
        testing.assert_allclose(st.vectors,nv)
        testing.assert_allclose(st.coordinates,
            ((0.,0.,0.),(2./3,0.,0.),))
        testing.assert_equal(st.values,('Co','Co'))
        
    def test_stack_fail_0(self):
        with self.assertRaises(ArgumentError):
            self.cell.stack(self.cell2, vector = 'z', restrict_collinear = True)
            
    def test_stack_fail_1(self):
        with self.assertRaises(ArgumentError):
            self.cell.stack(object(), vector = 'z')
        
    def test_repeated_0(self):
        rp = self.cell.repeated(1,1,1)
        testing.assert_allclose(rp.vectors,self.cell.vectors)
        testing.assert_allclose(rp.coordinates,self.cell.coordinates)
        testing.assert_equal(rp.values,self.cell.values)
        
    def test_repeated_1(self):
        rp = self.cell.repeated(2,2,1)
        nv = self.cell.vectors.copy()
        nv[0:2,:] *= 2
        testing.assert_allclose(rp.vectors,nv)
        testing.assert_allclose(rp.coordinates,((0.,0.,0.),
            (1./6,1./6,0.5),(.5,0.,0.),(2./3,1./6,0.5),
            (0.,0.5,0.),(1./6,2./3,0.5),(0.5,0.5,0),
            (2./3.,2./3,0.5)), atol = 1e-14)
        testing.assert_equal(rp.values,("Co",)*8)
        
    def test_select_0(self):
        testing.assert_equal(self.cell.select((0.,0.,0.,1.,1.,1.)),(True,True))
        
    def test_select_1(self):
        testing.assert_equal(self.cell.select((0.,0.,0.,.1,.1,.1)),(True,False))
        
    def test_select_2(self):
        testing.assert_equal(self.cell2.select((0.1,0.1,0.1,1.,1.,1.)),(False,))
        
    def test_select_fail(self):
        with self.assertRaises(ArgumentError):
            self.cell2.select(0,0,1,1)
        with self.assertRaises(ArgumentError):
            self.cell2.select(0,0,1,1,1)
        
    def test_apply(self):
        c = self.cell.copy()
        c.apply((False,True))
        testing.assert_equal(c.vectors,self.cell.vectors)
        testing.assert_allclose(c.coordinates,((1./3.,1./3.,0.5),))
        testing.assert_equal(c.values,('Co',))
        testing.assert_equal(c.values.shape, (1,))
        
    def test_discard_0(self):
        c = self.cell.copy()
        c.discard(c.select(1./6,1./6,0.,1.,1.,1.))
        testing.assert_equal(c.vectors,self.cell.vectors)
        testing.assert_allclose(c.coordinates,((0.,0.,0.),))
        testing.assert_equal(c.values,('Co',))
        
    def test_discard_1(self):
        c = self.cell.copy()
        c.discard((False, True))
        testing.assert_equal(c.vectors,self.cell.vectors)
        testing.assert_allclose(c.coordinates,((0.,0.,0.),))
        testing.assert_equal(c.values,('Co',))
        
    def test_normalized(self):
        c = self.cell.copy()
        c.coordinates += 0.6
        c.values[1] = 'OK'
        
        c1 = c.normalized()
        testing.assert_equal(c1.vectors,self.cell.vectors)
        testing.assert_allclose(c1.coordinates,((.6,.6,.6),(1./3 + .6,1./3 + .6,0.1)))
        testing.assert_equal(c1.values,('Co','OK'))
        
        c1 = c.normalized(sort = 'z')
        testing.assert_equal(c1.vectors,self.cell.vectors)
        testing.assert_allclose(c1.coordinates,((1./3 + .6,1./3 + .6,0.1),(.6,.6,.6)))
        testing.assert_equal(c1.values,('OK','Co'))
        
    def test_packed(self):
        cell = UnitCell(
            Basis(self.cell.vectors),
            coordinates = ((0.1,0.1,0),(0.9,0.1,0),(0.1,0.9,0),(0.9,0.9,0)),
            values = "A",
        ).packed()
        testing.assert_allclose(cell.coordinates,(
            (0.1,0.1,0),
            (-0.1,0.1,0),
            (0.1,-0.1,0),
            (-0.1,-0.1,0)
        ))
        
    def test_isolated_0(self):
        c = self.cell.isolated(1,2,3)
        nv = self.cell.vectors.copy()
        nv[0,:] *= 2
        nv[1,:] *= 3
        nv[2,:] *= 4
        testing.assert_allclose(c.vectors,nv)
        testing.assert_allclose(c.coordinates,((1./4,1./3,3./8),(5./12,4./9,1./2)))
        testing.assert_equal(c.values,('Co',)*2)
        
    def test_isolated_1(self):
        c = self.cell.isolated(self.a,2*self.a,3*self.h, units = 'cartesian')
        nv = self.cell.vectors.copy()
        nv[0,:] *= 2
        nv[1,:] *= 3
        nv[2,:] *= 4
        testing.assert_allclose(c.vectors,nv)
        testing.assert_allclose(c.coordinates,((1./4,1./3,3./8),(5./12,4./9,1./2)))
        testing.assert_equal(c.values,('Co',)*2)
        
    def test_isolated_fail(self):
        with self.assertRaises(ArgumentError):
            self.cell.isolated(1,2,3, units = 'unkown')
        
    def test_add(self):
        c = self.cell.copy()
        c.coordinates = .95-c.coordinates
        s = self.cell.add(c)
        testing.assert_allclose(s.coordinates,((0.,0.,0.),(1./3,1./3,.5),(.95,.95,.95),(.95-1./3,.95-1./3,.45)))
        testing.assert_equal(s.values,('Co',)*4)
        
    def test_add_fail(self):
        with self.assertRaises(ArgumentError):
            self.cell.add(self.cell2)
            
    def test_species_0(self):
        sp = self.cell.species()
        assert len(sp) == 1
        assert sp['Co'] == 2
        
    def test_species_1(self):
        c = self.cell.copy()
        c.values[0] = 'C'
        c = c.add(self.cell)
        sp = c.species()
        assert len(sp) == 2
        assert sp['Co'] == 3
        assert sp['C'] == 1
        
    def test_reorder_0(self):
        c = self.cell.copy()
        c.reorder_vectors(0,2,1)
        nv = self.cell.vectors.copy()
        nv = nv[(0,2,1),:]
        testing.assert_equal(c.vectors,nv)
        testing.assert_allclose(c.coordinates,((0.,0.,0.),(1./3,.5,1./3)))
        testing.assert_equal(c.values,('Co','Co'))
        
    def test_reorder_1(self):
        c = self.cell.copy()
        c.reorder_vectors('x','y','z')
        nv = self.cell.vectors.copy()
        testing.assert_equal(c.vectors,nv)
        testing.assert_allclose(c.coordinates,((0.,0.,0.),(1./3,1./3,.5)))
        testing.assert_equal(c.values,('Co','Co'))
        
    def test_pickle(self):
        c = pickle.loads(pickle.dumps(self.cell))
        testing.assert_equal(c.vectors, self.cell.vectors)
        testing.assert_equal(c.coordinates, self.cell.coordinates)
        testing.assert_equal(c.values, self.cell.values)
        
    def test_supercell(self):
        s = self.cell.supercell(
            (1,0,0),
            (-1,2,0),
            (0,0,1)
        )
        s.coordinates[:,0] += 0.25
        s.coordinates[:,1] += 1./6
        s = s.normalized(sort = 'y')
        
        testing.assert_allclose(s.vectors,(
            (self.a, 0, 0),
            (0,self.a*3.**.5,0),
            (0,0,self.h)
        ))
        testing.assert_allclose(s.coordinates,(
            (0.25,1./6,0.),
            (0.75,1./3,.5),
            (0.75,2./3,0.),
            (0.25,5./6,.5),
        ), atol = 1e-10)
        testing.assert_equal(s.values,('Co',)*4)
        
    def test_interpolate(self):
        c = UnitCell(
            Basis((1,1), kind = 'orthorombic', meta = {'key':'value'}),
            (
                (.5,.5),
                (0,0),
                (0,.5),
                (.5,0),
            ),
            ((1,5),(2,6),(3,7),(4,8))
        )
        
        for p in (True, False):
            
            c2 = c.interpolate((0,0),(.5,.5), periodic = p)
            
            testing.assert_equal(c.vectors, c2.vectors)
            testing.assert_equal(c.meta, c2.meta)
            testing.assert_equal(c2.coordinates, ((0,0),(.5,.5)))
            testing.assert_allclose(c2.values, ((2,6),(1,5)))
            
    def test_save_load(self):
        import numericalunits
        import pickle
        a = self.a*numericalunits.angstrom
        h = self.h*numericalunits.angstrom
        cell = CellTest.__co__(a,h,units = 'angstrom')
        assert cell.units_aware()
        
        data = pickle.dumps(cell)
        numericalunits.reset_units()
        x = pickle.loads(data)
        
        # Assert object changed
        assert x != cell
        
        # Assert object is the same wrt numericalunits
        a = self.a*numericalunits.angstrom
        h = self.h*numericalunits.angstrom
        cell2 = CellTest.__co__(a,h,units = 'angstrom')
        testing.assert_allclose(x.vectors,cell2.vectors,atol = 1e-8*numericalunits.angstrom)
        testing.assert_equal(x.coordinates,cell2.coordinates)
        testing.assert_equal(x.values,cell2.values)        
        
    def test_save_load_json(self):
        import json
        a = self.a*numericalunits.angstrom
        h = self.h*numericalunits.angstrom
        cell = CellTest.__co__(a,h,units = 'angstrom')
        assert cell.units_aware()
        
        data = json.dumps(cell.to_json())
        numericalunits.reset_units()
        x = UnitCell.from_json(json.loads(data))
        
        # Assert object changed
        assert x != cell
        
        # Assert object is the same wrt numericalunits
        a = self.a*numericalunits.angstrom
        h = self.h*numericalunits.angstrom
        cell2 = CellTest.__co__(a,h,units = 'angstrom')
        testing.assert_allclose(x.vectors,cell2.vectors,atol = 1e-8*numericalunits.angstrom)
        testing.assert_equal(x.coordinates,cell2.coordinates)
        testing.assert_equal(x.values,cell2.values)        

class FCCCellTest(unittest.TestCase):
    
    def test_sc_roundoff(self):
        si_basis = Basis((3.9/2, 3.9/2, 3.9/2, .5,.5,.5), kind = 'triclinic')
        si_cell = UnitCell(si_basis, (.5,.5,.5), 'Si')
        cubic_cell = si_cell.supercell(
            (1,-1,1),
            (1,1,-1),
            (-1,1,1),
        )
        testing.assert_allclose(
            cubic_cell.size()/cubic_cell.volume(),
            si_cell.size()/si_cell.volume(),
        )
            
class MultidimCellTest(unittest.TestCase):
    
    def setUp(self):
        self.c = UnitCell(
            Basis(numpy.eye(3)),
            (
                (0,0,0),
                (0,0,.1),
                (0,0,.2),
                (0,.1,.2),
            ),
            (
                (1,2),
                (2,3),
                (3,4),
                (4,5),
            )
        )
        
    def test_size(self):
        assert self.c.size() == 4

class Cell2Grid(unittest.TestCase):
    
    def test_as_grid(self):
        c = UnitCell(
            Basis((1,1), kind = 'orthorombic', meta = {'key':'value'}),
            (
                (.5,.5),
                (0,0),
                (0,.5),
                (.5,0),
            ),
            (1,2,3,4)
        )
        g = c.as_grid()
        
        testing.assert_equal(c.vectors, g.vectors)
        testing.assert_equal(c.meta, g.meta)
        assert len(g.coordinates) == 2
        testing.assert_equal(g.coordinates[0], (0,.5))
        testing.assert_equal(g.coordinates[1], (0,.5))
        testing.assert_equal(g.values, (
            (2,3),
            (4,1),
        ))
        
    def test_as_grid_missing(self):
        c = UnitCell(
            Basis((1,1), kind = 'orthorombic'),
            (
                (.5,.5),
                (0,0),
                (.5,0),
            ),
            (1,2,3)
        )
        g = c.as_grid()
        
        assert len(g.coordinates) == 2
        testing.assert_equal(g.coordinates[0], (0,.5))
        testing.assert_equal(g.coordinates[1], (0,.5))
        testing.assert_equal(g.values, (
            (2,float("nan")),
            (3,1),
        ))

class GridInitialiazationTest(unittest.TestCase):
                
    def test_init_grid_0(self):
        x = numpy.linspace(0,1,10, endpoint = False)
        y = numpy.linspace(0,1,20, endpoint = False)
        z = numpy.linspace(0,1,30, endpoint = False)
        xx,yy,zz = numpy.meshgrid(x,y,z,indexing='ij')
        data = xx*2+yy*2+zz*2
        meta = {"key":"value"}
        c = Grid(
            Basis((1,1,1,0,0,0), kind = 'triclinic', meta = meta),
            (x, y, z),
            data,
        )
        testing.assert_equal(c.meta, meta)
        assert len(c.coordinates) == 3
        testing.assert_equal(c.coordinates[0],x)
        testing.assert_equal(c.coordinates[1],y)
        testing.assert_equal(c.coordinates[2],z)
        testing.assert_equal(c.values.shape, (10,20,30))

    def test_init_grid_fail_0(self):
        x = numpy.linspace(0,1,10, endpoint = False)
        y = numpy.linspace(0,1,10, endpoint = False)
        z = numpy.linspace(0,1,10, endpoint = False)
        xx,yy,zz = numpy.meshgrid(x,y,z,indexing='ij')
        data = xx**2+yy**2+zz**2
        
        with self.assertRaises(ArgumentError):
            Grid(
                Basis((1,1,1,0,0,0), kind = 'triclinic'),
                numpy.array((x, y)),
                data,
            )
            
        with self.assertRaises(ArgumentError):
            Grid(
                Basis((1,1,1,0,0,0), kind = 'triclinic'),
                (x, y, z, z),
                data,
            )
            
        with self.assertRaises(ArgumentError):
            Grid(
                Basis((1,1,1,0,0,0), kind = 'triclinic'),
                ( (x,x), (y,y), (z,z)),
                data,
            )
            
        with self.assertRaises(ArgumentError):
            Grid(
                Basis((1,1,1,0,0,0), kind = 'triclinic'),
                ( x, y, z),
                data[0],
            )
            
    def test_init_grid_fail_1(self):
        x = numpy.linspace(0,1,10, endpoint = False)
        y = numpy.linspace(0,1,20, endpoint = False)
        z = numpy.linspace(0,1,30, endpoint = False)
        xx,yy,zz = numpy.meshgrid(x,y,z,indexing='ij')
        data = xx**2+yy**2+zz**2
            
        with self.assertRaises(ArgumentError):
            Grid(
                Basis((1,1,1,0,0,0), kind = 'triclinic'),
                ( x, x, x),
                data,
            )

class GridTest(unittest.TestCase):
    
    def setUp(self):
        x = numpy.linspace(0,1,2, endpoint = False)
        y = numpy.linspace(0,1,3, endpoint = False)
        z = numpy.linspace(0,1,4, endpoint = False)
        xx,yy,zz = numpy.meshgrid(x,y,z,indexing='ij')
        data = xx**2+yy**2+zz**2
        self.grid = Grid(
            Basis((1,2,3), kind = 'orthorombic'),
            (x, y, z),
            data,
        )
        self.empty = Basis((1,2,3), kind = 'orthorombic')
        
    def test_pickle(self):
        c = pickle.loads(pickle.dumps(self.grid))
        testing.assert_equal(c.vectors, self.grid.vectors)
        testing.assert_equal(c.coordinates, self.grid.coordinates)
        testing.assert_equal(c.values, self.grid.values)
        
    def test_pickle_units(self):
        import numericalunits
        grid = Grid(
            Basis(numpy.eye(3)*numericalunits.angstrom),
            self.grid.coordinates,
            self.grid.values,
            units='angstrom',
        )
        data = pickle.dumps(grid)
        numericalunits.reset_units()
        c = pickle.loads(data)
        testing.assert_equal(c.vectors, numpy.eye(3)*numericalunits.angstrom)
        testing.assert_equal(c.coordinates, grid.coordinates)
        testing.assert_equal(c.values, grid.values)
        
    def test_eq(self):
        g = self.grid.copy()
        assert self.grid == g
        g.coordinates[0][0] = 3.14
        assert not self.grid == g
        g.coordinates[0][0] = 0
        assert self.grid == g
        g.values[0,0,0] = 3.14
        assert not self.grid == g
        
    def test_size(self):
        assert self.grid.size() == 24

    def test_explicit_coordinates(self):
        c = self.grid.explicit_coordinates()
        assert len(c.shape) == 4
        assert (c[1,:,:,0] == 0.5).all()
        assert (c[:,1,:,1] == 1./3).all()
        assert (c[:,:,1,2] == 1./4).all()
        
    def test_cartesian(self):
        c = self.grid.cartesian()
        testing.assert_allclose(c[0,0,0], (0,0,0))
        testing.assert_allclose(c[1,2,3], (1./2*1, 2./3*2, 3./4*3))
        
    def test_select_0(self):
        testing.assert_equal(self.grid.select((0.3,0.3,0.3,0.7,0.7,0.7)),(
            (False, True),
            (False, True, True),
            (False, False, True, False)
        ))

    def test_select_1(self):
        testing.assert_equal(self.grid.select((0.3,0,0,0.7,1,1)),(
            (False, True),
            (True, True, True),
            (True, True, True, True)
        ))
        
    def test_select_fail(self):
        with self.assertRaises(ArgumentError):
            self.grid.select(0.3,0.3,0.7,0.7)

    def test_apply(self):
        c = self.grid.copy()
        c.apply((
            (False, True),
            (False, True, True),
            (False, False, True, False)
        ))
        testing.assert_allclose(c.coordinates[0], (0.5, ))
        testing.assert_allclose(c.coordinates[1], (1./3, 2./3 ))
        testing.assert_allclose(c.coordinates[2], (0.5, ))
        testing.assert_allclose(c.values, (
            ((0.5+1./9,), (0.5+4./9,)),
        ))
        
    def test_discard(self):
        c = self.grid.copy()
        c.discard((
            (False, True),
            (False, True, True),
            (False, False, True, False)
        ))
        testing.assert_allclose(c.coordinates[0], (0., ))
        testing.assert_allclose(c.coordinates[1], (0., ))
        testing.assert_allclose(c.coordinates[2], (0., .25, .75 ))
        testing.assert_allclose(c.values, (
            ((0., .25**2, .75**2),),
        ))
        
    def test_cut_0(self):
        c = self.grid.cut(0.3,0.3,0.3,0.7,0.7,0.7)
        testing.assert_allclose(c.coordinates[0], ((0.5-0.3)/0.4, ))
        testing.assert_allclose(c.coordinates[1], ((1./3-0.3)/0.4, (2./3-0.3)/0.4 ))
        testing.assert_allclose(c.coordinates[2], ((0.5-0.3)/0.4, ))
        testing.assert_allclose(c.values, (
            ((0.5+1./9,), (0.5+4./9,)),
        ))
        
    def test_cut_1(self):
        c = self.grid.cut(0.3,0,0,0.7,1,1)
        testing.assert_allclose(c.coordinates[0], ((0.5-0.3)/0.4, ))
        testing.assert_allclose(c.coordinates[1], self.grid.coordinates[1])
        testing.assert_allclose(c.coordinates[2], self.grid.coordinates[2])
        testing.assert_allclose(c.values, self.grid.values[[1],...])
        
    def test_add(self):
        x = numpy.linspace(0,1,2, endpoint = False)
        y = numpy.linspace(0,1,3, endpoint = False)
        z = numpy.linspace(0,1,4, endpoint = False)+0.1
        xx,yy,zz = numpy.meshgrid(x,y,z,indexing='ij')
        data = xx**2+yy**2+zz**2
        grid = Grid(
            Basis(self.grid.vectors),
            (x, y, z),
            data,
        )
        grid_merged = grid.add(self.grid)
        
        z = numpy.sort(numpy.concatenate((z, self.grid.coordinates[2])))
        xx,yy,zz = numpy.meshgrid(x,y,z,indexing='ij')
        data = xx**2+yy**2+zz**2
        grid_ref = Grid(
            Basis(self.grid.vectors),
            (x, y, z),
            data,
        )
        
        testing.assert_allclose(grid_merged.vectors, grid_ref.vectors)
        for i in range(3):
            testing.assert_allclose(grid_merged.coordinates[i], grid_ref.coordinates[i])
        testing.assert_allclose(grid_merged.values, grid_ref.values)

    def test_normalized(self):
        c = self.grid.copy()
        c.coordinates[0] += .1
        c.coordinates[1] += .2
        c.coordinates[2] += .3
        c = c.normalized()
        
        testing.assert_allclose(c.coordinates[0], (.1,.6))
        testing.assert_allclose(c.coordinates[1], (.2,1./3+.2,2./3+.2))
        testing.assert_allclose(c.coordinates[2], (.75-.7,.3,.25+.3,.5+.3))
        testing.assert_allclose(c.values, self.grid.values[:,:,(3,0,1,2)])

    def test_isolated_0(self):
        c = self.grid.isolated((1,2,3), units = 'crystal')

        testing.assert_allclose(c.coordinates[0], (self.grid.coordinates[0]+0.5)/2)
        testing.assert_allclose(c.coordinates[1], (self.grid.coordinates[1]+1)/3)
        testing.assert_allclose(c.coordinates[2], (self.grid.coordinates[2]+1.5)/4)
        testing.assert_allclose(c.values, self.grid.values)
        
    def test_isolated_1(self):
        c = self.grid.isolated((1,4,9), units = 'cartesian')

        testing.assert_allclose(c.coordinates[0], (self.grid.coordinates[0]+0.5)/2)
        testing.assert_allclose(c.coordinates[1], (self.grid.coordinates[1]+1)/3)
        testing.assert_allclose(c.coordinates[2], (self.grid.coordinates[2]+1.5)/4)
        testing.assert_allclose(c.values, self.grid.values)
        
    def test_isolated_fail(self):
        with self.assertRaises(ArgumentError):
            self.grid.isolated((1,2,3), units = 'unknown')
        
    def test_stack_0(self):
        another = self.grid.copy()
        another.vectors[2,2] = 6
        c = Grid.stack(self.grid, another, vector = 'z')

        testing.assert_allclose(c.coordinates[0], self.grid.coordinates[0])
        testing.assert_allclose(c.coordinates[1], self.grid.coordinates[1])
        testing.assert_allclose(c.coordinates[2], numpy.concatenate((
            self.grid.coordinates[2]/3, another.coordinates[2]*2/3+1./3
        )))
        testing.assert_allclose(c.values, numpy.concatenate((
            self.grid.values, another.values
        ), axis = 2))
        
    def test_stack_1(self):
        x = numpy.linspace(0,1,2, endpoint = False)
        y = numpy.linspace(0,1,3, endpoint = False)
        z = numpy.linspace(0.5,0.7,15)
        xx,yy,zz = numpy.meshgrid(x,y,z,indexing='ij')
        data = xx**2+yy**2+zz**2
        another = Grid(
            Basis((1,2,6), kind = 'orthorombic'),
            (x, y, z),
            data,
        )
        c = Grid.stack(self.grid, another, vector = 'z')

        testing.assert_allclose(c.coordinates[0], self.grid.coordinates[0])
        testing.assert_allclose(c.coordinates[1], self.grid.coordinates[1])
        testing.assert_allclose(c.coordinates[2], numpy.concatenate((
            self.grid.coordinates[2]/3, another.coordinates[2]*2/3+1./3
        )))
        testing.assert_allclose(c.values, numpy.concatenate((
            self.grid.values, another.values
        ), axis = 2))
        
    def test_stack_2(self):
        c = Grid.stack(self.grid, self.empty, vector = 'z')

        testing.assert_allclose(c.coordinates[0], self.grid.coordinates[0])
        testing.assert_allclose(c.coordinates[1], self.grid.coordinates[1])
        testing.assert_allclose(c.coordinates[2], self.grid.coordinates[2]/2)
        testing.assert_allclose(c.values, self.grid.values)
        
    def test_stack_error_0(self):
        another = UnitCell(
            Basis(self.grid.vectors),
            (0,0,0),
            'X',
        )
        with self.assertRaises(ArgumentError):
            self.grid.stack(another)
            
    def test_stack_error_1(self):
        x = numpy.linspace(0,1,3)[:-1]
        y = numpy.linspace(0,1,3)[:-1]
        z = numpy.linspace(0,1,5)[:-1]
        xx,yy,zz = numpy.meshgrid(x,y,z,indexing='ij')
        data = xx**2+yy**2+zz**2
        another = Grid(
            Basis(self.grid.vectors),
            (x, y, z),
            data,
        )
        with self.assertRaises(ArgumentError):
            self.grid.stack(another, vector = 'x')
       
    def test_repeated_0(self):
        rp = self.grid.repeated(1,1,1)
        testing.assert_allclose(rp.vectors,self.grid.vectors)
        assert len(rp.coordinates) == len(self.grid.coordinates)
        for i in range(len(rp.coordinates)):
            testing.assert_allclose(rp.coordinates[i],self.grid.coordinates[i])
        testing.assert_equal(rp.values,self.grid.values)
        
    def test_repeated_1(self):
        rp = self.grid.repeated(2,2,1)
        nv = self.grid.vectors.copy()
        nv[0:2,:] *= 2
        testing.assert_allclose(rp.vectors,nv)
        assert len(rp.coordinates) == len(self.grid.coordinates)
        
        testing.assert_allclose(rp.coordinates[0],numpy.linspace(0,1,4, endpoint = False))
        testing.assert_allclose(rp.coordinates[1],numpy.linspace(0,1,6, endpoint = False))
        testing.assert_allclose(rp.coordinates[2],numpy.linspace(0,1,4, endpoint = False))
        v = numpy.concatenate((self.grid.values,)*2, axis = 0)
        v = numpy.concatenate((v,)*2, axis = 1)
        testing.assert_equal(rp.values,v)
        
    def test_rv(self):
        c = self.grid.copy()
        c.reorder_vectors(2,1,0)

        testing.assert_allclose(c.coordinates[0], self.grid.coordinates[2])
        testing.assert_allclose(c.coordinates[1], self.grid.coordinates[1])
        testing.assert_allclose(c.coordinates[2], self.grid.coordinates[0])
        testing.assert_allclose(c.values, self.grid.values.swapaxes(0,2))
        
    def test_as_unitCell(self):
        reg = self.grid.as_unitCell()
        
        testing.assert_equal(reg.coordinates.shape, (24,3))
        testing.assert_equal(reg.values.shape, (24,))
        
        for i, x in enumerate(self.grid.coordinates[0]):
            for j, y in enumerate(self.grid.coordinates[1]):
                for k, z in enumerate(self.grid.coordinates[2]):
                    
                    found = False
                    
                    for l in range(reg.coordinates.shape[0]):
                        if tuple(reg.coordinates[l]) == (x,y,z):
                            found = True
                            assert self.grid.values[i,j,k] == reg.values[l]
                            break
                            
                    if not found:
                        raise AssertionError("Coordinate {} {} {} not found".format(x,y,z))

    def test_back_forth(self):
        c = self.grid.as_unitCell().as_grid()
        
        testing.assert_equal(c.coordinates, self.grid.coordinates)
        testing.assert_equal(c.values, self.grid.values)

    def test_interpolate_to_cell_0(self):
        i = self.grid.interpolate_to_cell(((0,0,0), (1./2,1./3,1./4)), periodic = False)

        testing.assert_equal(i.coordinates, ((0,0,0), (1./2,1./3,1./4)))
        testing.assert_equal(i.values, (0, 1./4+1./9+1./16))
        
    def test_interpolate_to_cell_1(self):
        i = self.grid.interpolate_to_cell(((0,0,0), (1./2,1./3,1./4), (-1./2, -2./3, -3./4)))

        testing.assert_equal(i.coordinates, ((0,0,0), (1./2,1./3,1./4), (-1./2, -2./3, -3./4)))
        testing.assert_equal(i.values, (0, 1./4+1./9+1./16, 1./4+1./9+1./16))
        
    def test_interpolate_to_cell_2(self):
        c = self.grid.copy()
        c.values = c.values[...,numpy.newaxis] * numpy.array((1,2))[numpy.newaxis, numpy.newaxis, numpy.newaxis, :]
        i = c.interpolate_to_cell(((0,0,0), (1./2,1./3,1./4)), periodic = False)
        
        testing.assert_equal(i.coordinates, ((0,0,0), (1./2,1./3,1./4)))
        testing.assert_equal(i.values, numpy.array((0, 1./4+1./9+1./16))[:,numpy.newaxis]*((1,2),))
        
    def test_interpolate_to_cell_3(self):
        c = self.grid.copy()
        c.values = c.values[...,numpy.newaxis] * numpy.array((1,2))[numpy.newaxis, numpy.newaxis, numpy.newaxis, :]
        i = c.interpolate_to_cell(((0,0,0), (1./2,1./3,1./4), (-1./2, -2./3, -3./4)))
        
        testing.assert_equal(i.coordinates, ((0,0,0), (1./2,1./3,1./4), (-1./2, -2./3, -3./4)))
        testing.assert_equal(i.values, numpy.array((0, 1./4+1./9+1./16, 1./4+1./9+1./16))[:,numpy.newaxis]*((1,2),))
        
    def test_interpolate_0(self):
        i = self.grid.interpolate_to_grid(self.grid.coordinates, periodic = False)

        testing.assert_equal(i.coordinates, self.grid.coordinates)
        testing.assert_equal(i.values, self.grid.values)

    def test_interpolate_error_0(self):
        with self.assertRaises(Exception):
            self.grid.interpolate(((0,0,0), (1./2,1./3,1./4), (-1./2, -2./3, -3./4)), periodic = False)
            
    def test_interpolate_periodic(self):
        x = numpy.linspace(0,1,2, endpoint = False)
        y = numpy.linspace(0,1,2, endpoint = False)
        z = numpy.linspace(0,1,2, endpoint = False)
        xx,yy,zz = numpy.meshgrid(x,y,z,indexing='ij')
        data = xx
        grid = Grid(
            Basis((1,2,3), kind = 'orthorombic'),
            (x, y, z),
            data,
        )
        interpolated = grid.interpolate_to_array(((.25,0,0),(.75,0,0)), periodic = True)
        testing.assert_equal(interpolated, (.25,.25))
            
    def test_interpolate_path_2D(self):
        x = numpy.linspace(0,1,10, endpoint = False)
        y = numpy.linspace(0,1,10, endpoint = False)
        xx,yy = numpy.meshgrid(x,y,indexing='ij')
        data = (xx-.5)**2+(yy-.5)**2
        grid = Grid(
            Basis((1,1), kind = 'orthorombic'),
            (x, y),
            data,
        )
        path = numpy.array((
            (0,0),
            (0,1),
            (1,0),
        ))
        cell = grid.interpolate_to_path(path,100)
        # Check values
        testing.assert_allclose(cell.values, ((cell.coordinates-.5)**2).sum(axis = -1), atol = 1e-2)
        
        # Check if all coordinates are on the path
        A = path[numpy.newaxis,:-1,:]
        B = path[numpy.newaxis, 1:,:]
        C = cell.coordinates[:,numpy.newaxis,:]
        Ax,Ay,Bx,By,Cx,Cy = A[...,0], A[...,1], B[...,0], B[...,1], C[...,0], C[...,1]
        area = Ax*(By-Cy) + Bx*(Cy-Ay) + Cx*(Ay-By)
        testing.assert_equal(numpy.any(area<1e-14, axis = 1), True)
        
        # Check if sum of spacings is equal to total path length
        assert abs(cell.distances(numpy.arange(cell.size())).sum()-1-2.**.5) < 1e-14
        
        # Check if spacings are uniform
        testing.assert_allclose(cell.distances(numpy.arange(cell.size())), (1+2.**.5) / (cell.size() - 1), rtol = 2./cell.size())
        
    def test_uniform(self):
        c = Grid.uniform((1,2,3))
        testing.assert_equal(c,[[
            [
                (0,0,0),
                (0,0,1./3),
                (0,0,2./3),
            ],[
                (0,.5,0),
                (0,.5,1./3),
                (0,.5,2./3),
            ],
        ]])

    def test_save_load_json(self):
        import json
        a = numericalunits.angstrom
        grid = Grid(Basis((a,2*a,3*a), kind='orthorombic'), self.grid.coordinates, self.grid.values, units = 'angstrom')
        assert grid.units_aware()
        
        data = json.dumps(grid.to_json())
        numericalunits.reset_units()
        x = Grid.from_json(json.loads(data))
        
        # Assert object changed
        assert x != grid
        
        # Assert object is the same wrt numericalunits
        a = numericalunits.angstrom
        grid2 = Grid(Basis((a,2*a,3*a), kind='orthorombic'), self.grid.coordinates, self.grid.values, units = 'angstrom')
        testing.assert_allclose(x.vectors,grid2.vectors,atol = 2e-8*numericalunits.angstrom)
        testing.assert_equal(x.coordinates,grid2.coordinates)
        testing.assert_equal(x.values,grid2.values)        

class TetrahedronDensityTest(unittest.TestCase):
    
    def setUp(self):
        N = 50
        x = numpy.linspace(-.5,.5,N, endpoint = False)
        y = numpy.linspace(-.5,.5,N, endpoint = False)
        z = numpy.array((0,))
        xx,yy,zz = numpy.meshgrid(x,y,z,indexing='ij')
        data = (xx**2+yy**2+zz**2)**.5
        self.grid = Grid(
            Basis((1,1,1), kind = 'orthorombic'),
            (x, y, z),
            data,
        )
        
    def test_td_0(self):
        d = self.grid.tetrahedron_density((-.1,0,.1,.2))
        testing.assert_allclose(d,(0,0,2*math.pi*0.1, 2*math.pi*0.2), rtol = 1e-2)
        
    def test_td_1(self):
        d = self.grid.tetrahedron_density((-.1,0,.1,.2), resolved = True)
        testing.assert_equal(d.values.shape, (50,50,1,1,4))
        testing.assert_allclose(d.values.sum(axis = 0).sum(axis = 0).sum(axis = 0)[0],(0,0,2*math.pi*0.1, 2*math.pi*0.2), rtol = 1e-2)

    def test_td_fail(self):
        g = self.grid.copy()
        g.vectors = g.vectors[:2,:2]
        g.coordinates = g.coordinates[:2]
        g.values = g.values[:,:,0,...]
        with self.assertRaises(ArgumentError):
            g.tetrahedron_density((-.1,0,.1,.2))
