# -*- coding: mbcs -*-

from odbAccess import openOdb
from textRepr import *
from abaqus import*
from abaqusConstants import*
from caeModules import *
import csv
import regionToolset
import time
from driverUtils import executeOnCaeStartup
import numpy as np

session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=317.6875, 
    height=205.333343505859)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()

executeOnCaeStartup()
openMdb('C:/Users/Free/Desktop/dataoutput_new/test.cae')
session.viewports['Viewport: 1'].setValues(displayedObject=None)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(referenceRepresentation=ON)


x = [9,9,9,9,9,9,9,9,9,     10,10]
z = [5,6,7,8,9,10,11,12,13, 5, 6]

for x_loc, z_loc in zip(x, z):

    start_time = time.time()
    parasolid = mdb.openParasolid(fileName='C:/Users/Free/Desktop/dataoutput_new/solid-object_new.x_t',
        topology=SOLID)
    mdb.models['Model-1'].PartFromGeometryFile(name='solid-object',
        geometryFile=parasolid, combine=False, dimensionality=THREE_D,
        type=DEFORMABLE_BODY, scale=1.0)
    p = mdb.models['Model-1'].parts['solid-object']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',
        sheetSize=200.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.rectangle(point1=(9.0, 0), point2=(-9.0, 1.5))
    p = mdb.models['Model-1'].Part(name='film', dimensionality=THREE_D,
        type=DEFORMABLE_BODY)
    p = mdb.models['Model-1'].parts['film']
    p.BaseSolidExtrude(sketch=s, depth=18.0)
    s.unsetPrimaryObject()
    p = mdb.models['Model-1'].parts['film']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models['Model-1'].sketches['__profile__']
    session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON,
        engineeringFeatures=ON)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=OFF)
    session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF,
        engineeringFeatures=OFF)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=ON)
    p1 = mdb.models['Model-1'].parts['solid-object']
    session.viewports['Viewport: 1'].setValues(displayedObject=p1)
    mdb.models['Model-1'].parts['solid-object'].setValues(space=THREE_D,
        type=DISCRETE_RIGID_SURFACE)
    p = mdb.models['Model-1'].parts['solid-object']
    c1 = p.cells
    p.RemoveCells(cellList = c1[0:1])

    session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON,
        engineeringFeatures=ON)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=OFF)
    mdb.models['Model-1'].Material(name='pdms')
    mdb.models['Model-1'].materials['pdms'].Hyperelastic(materialType=ISOTROPIC,
        testData=OFF, type=POLYNOMIAL, n=2, volumetricResponse=VOLUMETRIC_DATA,
        table=((-0.697, 1.08, 0.0, 2.52, 0.0, 0.0, 0.0), ))
    mdb.models['Model-1'].HomogeneousSolidSection(name='pdms-section',
        material='pdms', thickness=None)
    p = mdb.models['Model-1'].parts['film']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models['Model-1'].parts['film']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    region = p.Set(cells=cells, name='Set-film')
    p = mdb.models['Model-1'].parts['film']
    p.SectionAssignment(region=region, sectionName='pdms-section', offset=0.0,
        offsetType=MIDDLE_SURFACE, offsetField='',
        thicknessAssignment=FROM_SECTION)
    a = mdb.models['Model-1'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
    a = mdb.models['Model-1'].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    p = mdb.models['Model-1'].parts['film']
    a.Instance(name='film-1', part=p, dependent=ON)
    p = mdb.models['Model-1'].parts['solid-object']
    a.Instance(name='solid-object-1', part=p, dependent=ON)
    p1 = a.instances['solid-object-1']
    p1.translate(vector=(0.0, 0.0, 0.0))
    session.viewports['Viewport: 1'].view.fitView()
    a = mdb.models['Model-1'].rootAssembly
    a.translate(instanceList=('solid-object-1', ), vector=(-9.0, 2.901754, 0.0))

    a = mdb.models['Model-1'].rootAssembly
    a.translate(instanceList=('solid-object-1', ), vector=(x_loc, 0.0, z_loc))

    p = mdb.models['Model-1'].parts['film']
    f, e, d1 = p.faces, p.edges, p.datums
    t = p.MakeSketchTransform(sketchPlane=f[3], sketchUpEdge=e[10],
                              sketchPlaneSide=SIDE1, origin=(0.0, 0.5, 15.0))
    s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',
                                                 sheetSize=84.85, gridSpacing=2.12, transform=t)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=SUPERIMPOSE)
    p = mdb.models['Model-1'].parts['film']
    p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
    s1.rectangle(point1=(-13.5, 7.5), point2=(1.5, -7.5))
    p = mdb.models['Model-1'].parts['film']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#8 ]',), )
    e1, d2 = p.edges, p.datums
    p.PartitionFaceBySketch(sketchUpEdge=e1[10], faces=pickedFaces, sketch=s1)
    s1.unsetPrimaryObject()
    del mdb.models['Model-1'].sketches['__profile__']

    p = mdb.models['Model-1'].parts['solid-object']
    p.seedPart(size=0.6, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models['Model-1'].parts['solid-object']
    p.generateMesh()
    p = mdb.models['Model-1'].parts['film']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models['Model-1'].parts['film']
    p.seedPart(size=0.5, deviationFactor=0.1, minSizeFactor=0.1)
    elemType1 = mesh.ElemType(elemCode=C3D8RH, elemLibrary=STANDARD,
                              kinematicSplit=AVERAGE_STRAIN, hourglassControl=DEFAULT)
    elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
    elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD)
    p = mdb.models['Model-1'].parts['film']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]',), )
    pickedRegions = (cells,)
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2,
                                                       elemType3))
    p = mdb.models['Model-1'].parts['film']
    p.generateMesh()
    a1 = mdb.models['Model-1'].rootAssembly
    a1.regenerate()
    a = mdb.models['Model-1'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF,
                                                               adaptiveMeshConstraints=ON)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON,
                                                               adaptiveMeshConstraints=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=ON)

    contact_surf = []
    node_coordinates_label = []
    allNodes = mdb.models['Model-1'].rootAssembly.instances['film-1'].nodes
    for node in allNodes:
        temp = []
        if np.isclose(node.coordinates[1], 1.5):
            contact_surf.append(node.label - 1)
            temp.append(node.coordinates[0])
            temp.append(node.coordinates[1])
            temp.append(node.coordinates[2])
            temp.append(node.label)
            node_coordinates_label.append(temp)
    node_coordinates_label = np.array(node_coordinates_label)


    contact_surf = [mdb.models['Model-1'].rootAssembly.instances['film-1'].nodes[idx:idx + 1] for idx in contact_surf]
    mdb.models['Model-1'].rootAssembly.Set(name='Set-contact', nodes=contact_surf)

    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        adaptiveMeshConstraints=ON)
    mdb.models['Model-1'].StaticStep(name='Step-1', previous='Initial',
        maxNumInc=1000, initialInc=0.01, minInc=1e-15, nlgeom=ON)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')

    mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
        'S', 'PE', 'PEEQ', 'PEMAG', 'LE', 'U', 'RF', 'CF', 'CSTRESS', 'CDISP',
        'CFORCE'), timeInterval=0.01)
    mdb.models['Model-1'].ContactProperty('IntProp-1')
    mdb.models['Model-1'].interactionProperties['IntProp-1'].NormalBehavior(
        pressureOverclosure=HARD, allowSeparation=ON,
        constraintEnforcementMethod=DEFAULT)


    region1 = mdb.models['Model-1'].parts['solid-object'].Surface(name='m_Surf-1',
        side1Faces=mdb.models['Model-1'].parts['solid-object'].faces.getByBoundingBox(-20.0, -2.0, -20.0, 20.0, 10.0, 20.0))
    region2 = mdb.models['Model-1'].parts['film'].Surface(name='s_Surf-1',
        side1Faces=mdb.models['Model-1'].parts['film'].faces.findAt(((1.0, 1.5, 1.0),)))
    mdb.models['Model-1'].SurfaceToSurfaceContactStd(name='Int-1',
        createStepName='Step-1', main=mdb.models['Model-1'].rootAssembly.instances['solid-object-1'].surfaces['m_Surf-1'],
        secondary=mdb.models['Model-1'].rootAssembly.instances['film-1'].surfaces['s_Surf-1'],
        sliding=FINITE, thickness=ON, interactionProperty='IntProp-1', adjustMethod=NONE,
        initialClearance=OMIT, datumAxis=None, clearanceRegion=None)

    session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF,
        engineeringFeatures=OFF)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=ON)

    a1 = mdb.models['Model-1'].rootAssembly
    a1.regenerate()
    a = mdb.models['Model-1'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON,
        predefinedFields=ON, interactions=OFF, constraints=OFF,
        engineeringFeatures=OFF)
    a = mdb.models['Model-1'].rootAssembly
    f1 = a.instances['film-1'].faces
    faces1 = f1.getSequenceFromMask(mask=('[#10 ]', ), )
    region = a.Set(faces=faces1, name='Set-1')
    mdb.models['Model-1'].EncastreBC(name='BC-1', createStepName='Step-1',
        region=region, localCsys=None)
    a = mdb.models['Model-1'].rootAssembly
    f1 = a.instances['solid-object-1'].faces
    faces1 = f1.getSequenceFromMask(mask=('[#1 ]', ), )
    region = a.Set(faces=faces1, name='Set-2')
    mdb.models['Model-1'].DisplacementBC(name='BC-2', createStepName='Step-1',
        region=region, u1=0.0, u2=-1.5, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0,
        amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='',
        localCsys=None)
    a = mdb.models['Model-1'].rootAssembly
    f1 = a.instances['film-1'].faces
    faces1 = f1.getSequenceFromMask(mask=('[#4 ]', ), )
    a.Set(faces=faces1, name='Set-contact')
    p = mdb.models['Model-1'].parts['film']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models['Model-1'].parts['solid-object']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models['Model-1'].parts['solid-object']
    v1, e, d1, n = p.vertices, p.edges, p.datums, p.nodes
    p.ReferencePoint(point=p.InterestingPoint(edge=e[1], rule=CENTER))
    a1 = mdb.models['Model-1'].rootAssembly
    a1.regenerate()
    a = mdb.models['Model-1'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON, loads=OFF,
        bcs=OFF, predefinedFields=OFF, connectors=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=ON)
    p = mdb.models['Model-1'].parts['solid-object']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    session.viewports['Viewport: 1'].partDisplay.setValues(mesh=ON)
    session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
        meshTechnique=ON)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=OFF)


    i1 = mdb.models['Model-1'].rootAssembly.allInstances['solid-object-1']
    leaf = dgm.LeafFromInstance(instances=(i1, ))
    session.viewports['Viewport: 1'].assemblyDisplay.displayGroup.add(leaf=leaf)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=OFF)
    mdb.Job(name='Job-1', model='Model-1', description='', type=ANALYSIS,
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
        scratch='', resultsFormat=ODB, numThreadsPerMpiProcess=1,
        multiprocessingMode=DEFAULT, numCpus=8, numDomains=8, numGPUs=8)

    try:
        mdb.jobs['Job-1'].submit(consistencyChecking=OFF)
        mdb.jobs['Job-1'].waitForCompletion()

        odb = session.openOdb(name='Job-1.odb')
        variable = 'CNORMF'

        session.viewports['Viewport: 1'].makeCurrent()

        NodeSet = mdb.models['Model-1'].rootAssembly.sets['Set-contact']
        data_list = []
        datapoints = np.loadtxt('C:/Users/Free/Desktop/dataoutput_new/coordinates_labels.csv', delimiter=',')
        for i in range(101):
            frame_data = []
            for j in range(len(datapoints)):
                for iter, key in enumerate(odb.steps['Step-1'].frames[i].fieldOutputs.keys()):
                    if key == 'CNORMF   ASSEMBLY_FILM-1_S_SURF-1/ASSEMBLY_SOLID-OBJECT-1_M_SURF-1':  # contains 'Assembly ASSEMBLY' and 'Node TOPLINE-1.1'
                        three_components = np.array(odb.steps['Step-1'].frames[i].fieldOutputs[variable].values[j].data)
                        if len(three_components.shape) > 0:
                            data = np.array(odb.steps['Step-1'].frames[i].fieldOutputs[variable].values[j].data)[1]
                            frame_data.append(data)

            data_list.append(frame_data)

        data_list = np.array(data_list)
        np.savetxt("C:/Users/Free/Desktop/dataoutput_new/{}-{}-Y.csv".format(x_loc, z_loc), data_list, delimiter=',')
        print("{}-{}-Y.csv successfully!".format(x_loc, z_loc))

    except:
        pass

