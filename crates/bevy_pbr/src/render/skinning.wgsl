#define_import_path bevy_pbr::skinning

#import bevy_pbr::mesh_types::SkinnedMesh
#import bevy_pbr::mesh_bindings::mesh
#import bevy_render::maths::affine3_to_square

#ifdef SKINNED

#ifdef SKINS_USE_UNIFORM_BUFFERS
@group(2) @binding(1) var<uniform> joint_matrices: SkinnedMesh;
#else   // SKINS_USE_UNIFORM_BUFFERS
@group(2) @binding(1) var<storage> joint_matrices: array<mat4x4<f32>>;
#endif  // SKINS_USE_UNIFORM_BUFFERS

// An array of matrices specifying the joint positions from the previous frame.
//
// This is used for motion vector computation.
//
// If this is the first frame, or we're otherwise prevented from using data from
// the previous frame, this is simply the same as `joint_matrices` above.
#ifdef SKINS_USE_UNIFORM_BUFFERS
@group(2) @binding(6) var<uniform> prev_joint_matrices: SkinnedMesh;
#else   // SKINS_USE_UNIFORM_BUFFERS
@group(2) @binding(6) var<storage> prev_joint_matrices: array<mat4x4<f32>>;
#endif  // SKINS_USE_UNIFORM_BUFFERS

// The bit of a mesh's skin index that says its joint matrices are in the mesh's local space
// (a skin palette): the mesh's own transform then places the skinned vertex in the world.
const LOCAL_SPACE_SKIN_INDEX_BIT: u32 = 1u << 31u;

fn skin_model(
    indexes: vec4<u32>,
    weights: vec4<f32>,
    instance_index: u32,
) -> mat4x4<f32> {
    let raw_skin_index = mesh[instance_index].current_skin_index;
#ifdef SKINS_USE_UNIFORM_BUFFERS
    let skinned = weights.x * joint_matrices.data[indexes.x]
        + weights.y * joint_matrices.data[indexes.y]
        + weights.z * joint_matrices.data[indexes.z]
        + weights.w * joint_matrices.data[indexes.w];
#else   // SKINS_USE_UNIFORM_BUFFERS
    let skin_index = raw_skin_index & ~LOCAL_SPACE_SKIN_INDEX_BIT;
    let skinned = weights.x * joint_matrices[skin_index + indexes.x]
        + weights.y * joint_matrices[skin_index + indexes.y]
        + weights.z * joint_matrices[skin_index + indexes.z]
        + weights.w * joint_matrices[skin_index + indexes.w];
#endif  // SKINS_USE_UNIFORM_BUFFERS
    if (raw_skin_index & LOCAL_SPACE_SKIN_INDEX_BIT) != 0u && raw_skin_index != 0xffffffffu {
        return affine3_to_square(mesh[instance_index].world_from_local) * skinned;
    }
    return skinned;
}

// Returns the skinned position of a vertex with the given weights from the
// previous frame.
//
// This is used for motion vector computation.
fn skin_prev_model(
    indexes: vec4<u32>,
    weights: vec4<f32>,
    instance_index: u32,
) -> mat4x4<f32> {
    let raw_skin_index = mesh[instance_index].current_skin_index;
#ifdef SKINS_USE_UNIFORM_BUFFERS
    let skinned = weights.x * prev_joint_matrices.data[indexes.x]
        + weights.y * prev_joint_matrices.data[indexes.y]
        + weights.z * prev_joint_matrices.data[indexes.z]
        + weights.w * prev_joint_matrices.data[indexes.w];
#else   // SKINS_USE_UNIFORM_BUFFERS
    let skin_index = raw_skin_index & ~LOCAL_SPACE_SKIN_INDEX_BIT;
    let skinned = weights.x * prev_joint_matrices[skin_index + indexes.x]
        + weights.y * prev_joint_matrices[skin_index + indexes.y]
        + weights.z * prev_joint_matrices[skin_index + indexes.z]
        + weights.w * prev_joint_matrices[skin_index + indexes.w];
#endif  // SKINS_USE_UNIFORM_BUFFERS
    if (raw_skin_index & LOCAL_SPACE_SKIN_INDEX_BIT) != 0u && raw_skin_index != 0xffffffffu {
        return affine3_to_square(mesh[instance_index].previous_world_from_local) * skinned;
    }
    return skinned;
}

fn inverse_transpose_3x3m(in: mat3x3<f32>) -> mat3x3<f32> {
    let x = cross(in[1], in[2]);
    let y = cross(in[2], in[0]);
    let z = cross(in[0], in[1]);
    let det = dot(in[2], z);
    return mat3x3<f32>(
        x / det,
        y / det,
        z / det
    );
}

fn skin_normals(
    world_from_local: mat4x4<f32>,
    normal: vec3<f32>,
) -> vec3<f32> {
    return normalize(
        inverse_transpose_3x3m(
            mat3x3<f32>(
                world_from_local[0].xyz,
                world_from_local[1].xyz,
                world_from_local[2].xyz
            )
        ) * normal
    );
}

#endif
