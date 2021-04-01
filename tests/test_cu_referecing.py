#!/usr/bin/python3
import unittest
from unittest import TestCase, mock, skip
from xrtm_decoder import iter_cu_references

class TestCuReferencing2(TestCase):

    def test_grid_boundaries(self):

        # reference all CUs arround the current position
        coded_refs = int('0b11111111', 2)

        ##############################################################
        # topleft
        X = 0
        expected = [
            X,    X+1,
            X+32, X+33
        ]
        refs = [*iter_cu_references(coded_refs, X, 32, 32)]
        self.assertEqual(len(refs), len(expected))
        for a in expected:
            self.assertIn(a, refs)
        
        ###############
        # top
        X = int(32/2)
        expected = [
            X-1,  X,    X+1,
            X+31, X+32, X+33
        ]
        refs = [*iter_cu_references(coded_refs, X, 32, 32)]
        self.assertEqual(len(refs), len(expected))
        for a in expected:
            self.assertIn(a, refs)

        ###############
        # topright
        X = 32 - 1
        expected = [
            X-1,  X,
            X+31, X+32,
        ]
        refs = [*iter_cu_references(coded_refs, X, 32, 32)]
        self.assertEqual(len(refs), len(expected))
        for a in expected:
            self.assertIn(a, refs)
        
        ##############################################################
        # left
        X = int(32/2)*32
        expected = [
            X-32, X-32, 
            X,    X+1,
            X+32, X+33
        ]
        refs = [*iter_cu_references(coded_refs, X, 32, 32)]
        self.assertEqual(len(refs), len(expected))
        for a in expected:
            self.assertIn(a, refs)

        ###############
        # center
        X = int(32/2)*32 + int(32/2)
        expected = [
            X-33, X-32, X-31,
            X-1,  X,    X+1,
            X+31, X+32, X+33
        ]
        refs = [*iter_cu_references(coded_refs, X, 32, 32)]
        self.assertEqual(len(refs), len(expected))
        for a in expected:
            self.assertIn(a, refs)

        ###############
        # right
        X = int(32/2)*32 - 1
        expected = [
            X-33, X-32,
            X-1,  X,
            X+31, X+32
        ]
        refs = [*iter_cu_references(coded_refs, X, 32, 32)]
        self.assertEqual(len(refs), len(expected))
        for a in expected:
            self.assertIn(a, refs)
        
        ##############################################################
        # bottomleft
        X = 31*32
        expected = [
            X-32, X-31,
            X,    X+1
        ]
        refs = [*iter_cu_references(coded_refs, X, 32, 32)]
        self.assertEqual(len(refs), len(expected))
        for a in expected:
            self.assertIn(a, refs)

        ###############
        # bottom
        X = 31*32 + int(32/2)
        expected = [
            X-33, X-32, X-31,
            X-1,  X,    X+1
        ]
        refs = [*iter_cu_references(coded_refs, X, 32, 32)]
        self.assertEqual(len(refs), len(expected))
        for a in expected:
            self.assertIn(a, refs)

        ###############
        # bottomright
        X = 32*32 - 1
        expected = [
            X-33, X-32,
            X-1,  X
        ]
        refs = [*iter_cu_references(coded_refs, X, 32, 32)]
        self.assertEqual(len(refs), len(expected))
        for a in expected:
            self.assertIn(a, refs)


if __name__ == '__main__':
    unittest.main()