import bpy
import bmesh
import numpy as np
import mathutils as mu

# info about this plugin
bl_info = {
    "name": "Regularization",
    "author": "Naoya Tsuruta",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "Mesh > Regularization",
    "description": "Equilateral triangulation.",
    "warning": "",
    "support": "TESTING",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Mesh"
}

class OBJECT_OT_Regularization(bpy.types.Operator):
    bl_idname = "object.regularization" # ID
    bl_label = "Regularize" # menu label
    bl_description = "Regularize mesh" # description
    bl_options = {'REGISTER', 'UNDO'} # attributes
	
    DELTA = 1.0e-7
    
    # initialize matrices and variables
    def init(self, me):
        #
        NUM_CONSTRAINTS = 1
        #
        self.mesh = me
        vn = len(me.verts)
        vn3 = vn * 3
        en = len(me.edges)
        
        self.A = np.zeros((vn3, vn3))
        self.hessian = np.zeros((vn3, vn3))
        self.jacobian = np.zeros((NUM_CONSTRAINTS, vn3))
        self.fx = np.zeros(NUM_CONSTRAINTS)
        self.d_fx = np.zeros(vn3)
        self.dir = np.zeros(vn3)
        self.tmp = np.zeros(NUM_CONSTRAINTS)
        self.elu = np.zeros(en)
        self.lam = 1
        self.backup = np.zeros((3, vn))
        
        self.NV = vn
        self.NV2 = vn * 2
        self.NV3 = vn3
        self.RANGE_NV = range(vn)
        self.RANGE_NV3 = range(vn3)
        self.RANGE_CONSTRAINTS = range(NUM_CONSTRAINTS)

        # for presice deformation
        if meanEdgeLength(me.edges)<OBJECT_OT_Regularization.DELTA:
            OBJECT_OT_Regularization.DELTA *= 1e-1
        
    def cost(self):
        self.computeEdgeLengthUniqueness()
        self.fx[0] = np.sum(self.elu)
        return np.sum(self.fx)
    
    def costAt(self, i):
        self.computeEdgeLengthUniquenessAt(i)
        self.fx[0] = np.sum(self.elu)
        return np.sum(self.fx)
    
    def computeNumericalJacobian(self):
        invDelta = 1.0/OBJECT_OT_Regularization.DELTA
        fx_tmp = -invDelta * self.cost()
        np.copyto(self.tmp, self.fx)
        self.tmp *= -invDelta
        # views
        d_fx_y = self.d_fx[self.NV:self.NV2]
        d_fx_z = self.d_fx[self.NV2:self.NV3]
        jacobian_y = self.jacobian[:,self.NV:self.NV2]
        jacobian_z = self.jacobian[:,self.NV2:self.NV3]

        for v in self.mesh.verts:
            p = v.co
            i = v.index
            # slightly moves vertex along x-axis and compute jacobian
            p.x += OBJECT_OT_Regularization.DELTA
            self.d_fx[i] = self.costAt(i)*invDelta + fx_tmp
            self.fx *= invDelta
            self.fx += self.tmp
            for j in self.RANGE_CONSTRAINTS:
                self.jacobian[j,i] = self.fx[j]
            p.x -= OBJECT_OT_Regularization.DELTA
            # y-axis
            p.y += OBJECT_OT_Regularization.DELTA
            d_fx_y[i] = self.costAt(i)*invDelta + fx_tmp
            self.fx *= invDelta
            self.fx += self.tmp
            for j in self.RANGE_CONSTRAINTS:
                jacobian_y[j,i] = self.fx[j]
            p.y -= OBJECT_OT_Regularization.DELTA
            # z-axis
            p.z += OBJECT_OT_Regularization.DELTA
            d_fx_z[i] = self.costAt(i)*invDelta + fx_tmp
            self.fx *= invDelta
            self.fx += self.tmp
            for j in self.RANGE_CONSTRAINTS:
                jacobian_z[j,i] = self.fx[j]
            p.z -= OBJECT_OT_Regularization.DELTA
            #
            self.costAt(i)
        
    def computeA(self, la):
        np.copyto(self.A, self.hessian)
        for i in self.RANGE_NV3:
            self.A[i,i] += la
    
    def computeEdgeLengthUniqueness(self):
        for ed in self.mesh.edges:
            dif = 1.0 - length(ed)
            self.elu[ed.index] = dif * dif
    
    # compute edge-lengths around given vertex
    def computeEdgeLengthUniquenessAt(self, i):
        links = self.mesh.verts[i].link_edges
        for ed in links:
            dif = 1.0 - length(ed)
            self.elu[ed.index] = dif * dif
    
    def gaussSeidel(self, M, v, b):
        numRows = len(M[0])
        diag = 0
        for i in range(numRows):
            curr = M[i]
            diag = curr[i]
            v[i] += (b[i]-np.dot(curr,v))/diag
    
    def backupVertsPositions(self):
        i = 0
        for v in self.mesh.verts:
            self.backup[0,i] = v.co.x
            self.backup[1,i] = v.co.y
            self.backup[2,i] = v.co.z
            i += 1
    
    def restoreVertsPositions(self):
        i = 0
        for v in self.mesh.verts:
            v.co.x = self.backup[0,i]
            v.co.y = self.backup[1,i]
            v.co.z = self.backup[2,i]
            i += 1
    
    def regularize(self, itr):
        print("start")
        cost = self.cost()
        prevCost = cost
        dif = 1e+10 # sys.float_info.max
        
        i = 0
        self.lam = 1.0
        MAX_ITERATIONS = 10000
        RANGE_5 = range(5)
        # views
        dir_y = self.dir[self.NV:self.NV2]
        dir_z = self.dir[self.NV2:self.NV3]
        
        for i in range(MAX_ITERATIONS):
            self.computeNumericalJacobian()
            self.hessian = self.jacobian.transpose() * self.jacobian
            
            foundBetter = False
            for k in RANGE_5:
                self.computeA(self.lam)
                self.gaussSeidel(self.A, self.dir, self.d_fx)
                # save positions of all vertices
                self.backupVertsPositions()
                # move vertices
                for v in self.mesh.verts:
                    i = v.index
                    v.co.x -= self.dir[i]
                    v.co.y -= dir_y[i]
                    v.co.z -= dir_z[i]
                
                cost = self.cost()
                if cost < prevCost:
                    # smaller solution have found
                    foundBetter = True
                    dif = prevCost - cost
                    prevCost = cost
                    self.lam *= 0.1
                else:
                    self.lam *= 10.0
                    self.restoreVertsPositions()
            
            if foundBetter == False:
                break
            if dif < 1e-15:
                break
    
    def execute(self, context):
        #me = bpy.context.object.data
        selections = bpy.context.selected_objects
        if not selections:
            return {'CANCELLED'}
        me = selections[0].data
        # create new bmesh
        bm = bmesh.new()
        bm.from_mesh(me)
        # enable lookup
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        # normalize entire object
        normalize(bm)
        # regularization
        self.init(bm)
        self.regularize(100) # 100 loops for test
        # update mesh
        bm.to_mesh(me)
        # result
        self.report({'INFO'}, "mean_length:{:.10e}".format(meanEdgeLength(self.mesh.edges)))
        #print ("mean_length:{:.10e}".format(meanEdgeLength(bm.edges)))
        # refresh viewport
        #layer = bpy.context.view_layer
        #layer.update()
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        return {'FINISHED'}

# menu panel
class PANEL_PT_Regularization(bpy.types.Panel):
    bl_label = "Regularization"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"
    
    def draw(self, context):
        self.layout.operator("object.regularization") # create a button

classes = [
    OBJECT_OT_Regularization,
    PANEL_PT_Regularization
]

# register & unregister
register, unregister = bpy.utils.register_classes_factory(classes)


# length of an edge
def length(ed):
    sv = ed.verts[0].co
    ev = ed.verts[1].co
    return distance(sv, ev)
def distance(a, b):
	return np.sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z))

# mean length of edges
def meanEdgeLength(edges):
    sum = 0
    for ed in edges:
        sum += length(ed)
    return sum / len(edges)

# normalize
def normalize(me):
    s = np.sqrt(1.0 / meanEdgeLength(me.edges))
    for v in me.verts:
        v.co *= s

# main
if __name__ == "__main__":
    register()
