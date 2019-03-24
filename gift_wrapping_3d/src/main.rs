extern crate image;
#[macro_use]
extern crate vulkano_shader_derive;
#[macro_use]
extern crate vulkano;
extern crate vulkano_win;
extern crate winit;
extern crate vulkano_glyph;
extern crate cgmath;
extern crate time;
extern crate rand;

/* this program will produce the convex hull for a set of 15 points using the gift
    wrapping algorithm. press the space bar once rendering and watch the hull appear

    drew fletcher

*/


use vulkano::instance::Instance;
use vulkano::instance::PhysicalDevice;
use vulkano::device::Device;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::framebuffer::Subpass;
use vulkano::framebuffer::Framebuffer;
use std::sync::Arc;
use vulkano::swapchain;
use vulkano::swapchain::{AcquireError, SwapchainCreationError, Swapchain, SurfaceTransform, PresentMode};
use vulkano::command_buffer::DynamicState;
use vulkano::pipeline::viewport::Viewport;
use vulkano_win::VkSurfaceBuild;
use winit::EventsLoop;
use winit::WindowBuilder;
use vulkano::sync::GpuFuture;
use vulkano::sync::now;
use std::time::{Duration, Instant};
use std::vec::Vec;
use rand::Rng;
    //vertex class; this is what is passed to rendering pipeline
    #[derive(Debug, Copy, Clone)]
    struct Vertex {
        position: [f32; 3],
        v_color: [f32; 4],
        index: u32,
    }   
    impl_vertex!(Vertex, position, v_color, index);

    //EDGE CLASS
    #[derive(Debug, Copy, Clone)] 
    struct Edge {
        point1: Vertex,
        point2: Vertex,
        processed: bool,
    }

    impl Edge {
        fn set_true(&mut self) {
            self.processed = true;
        }
    }
    //Triangle class
    #[derive(Debug, Copy, Clone)]
    struct Triangle {
        point1: Vertex,
        point2: Vertex,
        point3: Vertex,
    }

fn main() {

   // let mut p_set = Vec::new();
    let mut p_set = make_some_points();
    
    //ignore this
    const INDICES:  [u32; 43] = [
        //front
        0, 2, 6,

        0, 4, 6,

        //left 

        0, 3, 2,

        1, 0, 3,

        //top

        0, 5, 1,

        0, 5, 4, 

        //right

        4, 7, 6,

        4, 7, 5,

        //back

        5, 1, 7,

        1, 3, 5,

        //bottom

        2, 3, 6,

        3, 6, 7,

        8, 9, 10,

        11, 12, 13,

        14,
    ];
    //ignore this as well
    let mut line_idx : [u32; 2] = [
        9, 8
    ];
    //call gift wrapping method, returns set of triangles
    let H = gift_wrap_3(&p_set);

    let size = H.len();


    for i in 0 .. p_set.len() {
        println!( "{} = {:?}", i, p_set[i]);
    }

    let mut arr = vec![0; (size)*3];
    let mut j = 0;
    for i in 0..size {
        //println!("{:?}", H[i]);
        arr[j] = H[i].point1.index;
        arr[j+1] = H[i].point2.index;
        arr[j+2] = H[i].point3.index;
        j+=3;
        //in this loop, we put each vertex into a vector that will be loaded into the vertex buffer
        //println!("{:?}", H[i]);
    }

    println!("size is: {}", arr.len());
    for i in 0..arr.len() {
        println!("{}", arr[i]);
    }


    //get device extensions for GPU
    let instance = {
        let winit_extens = vulkano_win::required_extensions();
        Instance::new(None, &winit_extens, None)
        .expect("failed to create instance")
    };

    //grab object to physical GPU device
    let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");
    for family in physical.queue_families() {
        println!("Found a queue family with {:?} queue(s)", family.queues_count());
    }

    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let mut events_loop = EventsLoop::new();
    
    //create event loop and window for rendering
    let surface = WindowBuilder::new()
    .with_title("Convex Hull")
    .build_vk_surface(&events_loop, instance.clone()).unwrap();

    //message queue to send commands to GPU
    let queue_family = physical.queue_families()
        .find(|&q| {
            q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
        }).expect("couldn't find a graphical queue family");

    //create an interface to GPU device and register a queue to send commands
    let (device, mut queues) = {

        let device_ext = vulkano::device::DeviceExtensions {
            khr_swapchain: true,
            .. vulkano::device::DeviceExtensions::none()
        };

        Device::new(physical, physical.supported_features(), &device_ext, 
        [(queue_family, 0.5)].iter().cloned()).expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    let mut dimensions;
    //create swapchain, which manages which images are being rendered
    let (mut swapchain, mut images) = {
        
        let caps = surface.capabilities(physical)
                         .expect("failed to get surface capabilities");

        dimensions = caps.current_extent.unwrap_or([1024, 768]);

        
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        
        let format = caps.supported_formats[0].0;

        Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format,
                       dimensions, 1, caps.supported_usage_flags, &queue,
                       SurfaceTransform::Identity, alpha, PresentMode::Fifo, true,
                       None).expect("failed to create swapchain")
    };

    //create depth buffer, which measures which pixels to block when rendering
    let mut depth_buffer = vulkano::image::attachment::AttachmentImage::transient(device.clone(), dimensions, vulkano::format::D16Unorm).unwrap();





    /*let ver_buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), 
                            vec![v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14].into_iter()).unwrap();*/
    //create buffers for rendering
    let ver_buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), 
                            p_set.into_iter()).unwrap();

    let index_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer
                                ::from_iter(device.clone(), vulkano::buffer::BufferUsage::all(), INDICES.iter().cloned())
                                .expect("failed to create buffer");

    let idx = vulkano::buffer::cpu_access::CpuAccessibleBuffer
                                ::from_iter(device.clone(), vulkano::buffer::BufferUsage::all(), arr.iter().cloned())
                                .expect("failed to create buffer");

    //create projection matrix to start creating our world space
    let mut proj = cgmath::perspective(cgmath::Rad(std::f32::consts::FRAC_PI_2), { dimensions[0] as f32 / dimensions[1] as f32 }, 0.01, 100.0);

    let mut view_x = 0.0;
    //view determines direction of camera and scalar matrix
    let mut view = cgmath::Matrix4::look_at(cgmath::Point3::new(view_x, 0.3, 1.0), cgmath::Point3::new(0.0, 0.0, 0.0), cgmath::Vector3::new(0.0, -1.0, 0.0));
    let scale = cgmath::Matrix4::from_scale(0.01);

    //create buffer to pass data to the shaders
    let uniform_buffer = vulkano::buffer::cpu_pool::CpuBufferPool::<vs::ty::Data>
                               ::new(device.clone(), vulkano::buffer::BufferUsage::all());


    let mut mBoxTrans = cgmath::Matrix4::from_translation(cgmath::Vector3::new(1.0, -15.0, -5.0));

    let mut mBoxScale = cgmath::Matrix4::from_nonuniform_scale(1.0, 1.0, 1.0);

    //shader for each vertex
    mod vs {
    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[src = "
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 v_color;
layout(location = 2) out vec4 color;
layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;
void main() {
    mat4 worldview = uniforms.view * uniforms.world;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
    gl_PointSize = 5.0; 
    color = v_color;
}
"]
    #[allow(dead_code)]
    struct Dummy;
}
    //fragment shader for the faces of the hull
    mod fs {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[src = "
#version 450

layout(location = 0) out vec4 f_color;
layout(location = 2) in vec4 color;

void main() {
    f_color = color;
}"  
    ]
    struct place_holder;
}
    //fragment shader for the points in space
   mod fs_2 {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[src = "
#version 450

layout(location = 0) out vec4 f_color;
layout(location = 2) in vec4 color;

void main() {
    f_color = vec4(0.0, 0.0, 0.0, 1.0);
}"  
    ]
    struct place_holder;
}
    
    let vs = vs::Shader::load(device.clone()).expect("Failed to create vertex shader");
    let fs = fs::Shader::load(device.clone()).expect("Failed to create vertex shader");
    let fs_2 = fs_2::Shader::load(device.clone()).expect("Failed to create vertex shader");

    //the render pass is the settings for each stage of the pipeline
    let render_pass = Arc::new(single_pass_renderpass!(device.clone(),
        attachments: {
        
            color: {
   
                load: Clear,
                
                store: Store,
                
                format: swapchain.format(),
                
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: vulkano::format::Format::D16Unorm,
                samples: 1,
            }
        },
        pass: {
            
            color: [color],
        
            depth_stencil: {depth}
        }
    ).unwrap());

    let mut change_pip = true;
    
    let mut time_stamp = Instant::now();
    
    //build the actual pipelone object
    let pipeline_main = Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .depth_stencil_simple_depth()
            
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap()); 
    
    
    let mut framebuffers: Option<Vec<Arc<vulkano::framebuffer::Framebuffer<_,_>>>> = None;

    let mut recreate_swapchain = false;

    let mut previous_frame_end = Box::new(now(device.clone())) as Box<GpuFuture>;

    let rotation_start = std::time::Instant::now();

    let mut cam_x_pos = view_x;


    //create the viewport for the window
    let mut dynamic_state = DynamicState {
        line_width: None,
        viewports: Some(vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            depth_range: 0.0 .. 1.0,
        }]),
        scissors: None,
    };

    let mut arr2 = vec![0; size*3];

    let mut j = 0 as usize;

    let mut i = 0 as usize;

    let mut done = false;

    let mut show = false;

    loop {
        
        previous_frame_end.cleanup_finished();

        //necessary to recreate the swapchain for each frame
        if recreate_swapchain {
            
            dimensions = surface.capabilities(physical)
                        .expect("failed to get surface capabilities")
                        .current_extent.unwrap();

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                
                Err(SwapchainCreationError::UnsupportedDimensions) => {
                    continue;
                },
                Err(err) => panic!("{:?}", err)
            };

            swapchain = new_swapchain;
            images = new_images;

            depth_buffer = vulkano::image::attachment::AttachmentImage::transient(device.clone(), dimensions, vulkano::format::D16Unorm).unwrap();

            framebuffers = None;

            proj = cgmath::perspective(cgmath::Rad(std::f32::consts::FRAC_PI_2), { dimensions[0] as f32 / dimensions[1] as f32 }, 0.01, 100.0);

            dynamic_state.viewports = Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                depth_range: 0.0 .. 1.0,
            }]);

            recreate_swapchain = false;
        }

       
        if framebuffers.is_none() {
            framebuffers = Some(images.iter().map(|image| {
                Arc::new(Framebuffer::start(render_pass.clone())
                         .add(image.clone()).unwrap()
                         .add(depth_buffer.clone()).unwrap()
                         .build().unwrap())
            }).collect::<Vec<_>>());
        }

        

        //send data about world space to the GPU
        let uniform_buffer_subbuffer = {


            let elapsed = rotation_start.elapsed();
            let rotation = elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
            let rotation = cgmath::Matrix3::from_angle_y(cgmath::Rad(rotation as f32));
            
            let mut world_view = cgmath::Matrix4::from(rotation);

            mBoxTrans = cgmath::Matrix4::from_translation(cgmath::Vector3::new(cam_x_pos, 5.0, -25.0));

            world_view = world_view * mBoxScale * mBoxTrans;
            




            let uniform_data = vs::ty::Data {
                world : world_view.into(),
                view : (view * scale).into(),
                proj : proj.into(),
            };

            uniform_buffer.next(uniform_data).unwrap()
        };
        
        
         let po_pipeline = Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs_2.main_entry_point(), ())
            .depth_stencil_simple_depth()
            .point_list()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap());
       
    
        

        let line_pip = Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .depth_stencil_simple_depth()
            .line_list()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap()); 

        let set = Arc::new(vulkano::descriptor::descriptor_set::PersistentDescriptorSet::start(pipeline_main.clone(), 0)
                .add_buffer(uniform_buffer_subbuffer).unwrap()
                .build().unwrap()
        );
           

        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(),
                                                                              None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                continue;
            },
            Err(err) => panic!("{:?}", err)
        };


        let elapsed = Instant::now();

        

        if elapsed.duration_since(time_stamp).as_secs() as f64 > 1.0/3.0 as f64 {

            //load the array passed to the vertex buffer with the faces of the hull
            if show {
                for i in 0 .. arr.len() {
                        arr2[i] = arr[i];
                    }
                    show = false;
            }

            if line_idx[1] as i32 == 0 {
                line_idx[1] = 8;
            } else {
                line_idx[1]-=1;
            }
            time_stamp = elapsed;
        }

        let index_buffer_line = vulkano::buffer::cpu_access::CpuAccessibleBuffer
                                ::from_iter(device.clone(), vulkano::buffer::BufferUsage::all(), line_idx.iter().cloned())
                                .expect("failed to create buffer");
        let index_buffer_tri = vulkano::buffer::cpu_access::CpuAccessibleBuffer
                                ::from_iter(device.clone(), vulkano::buffer::BufferUsage::all(), arr2.iter().cloned())
                                .expect("failed to create buffer");
        //send actual commands to the GPU, this is when the vertices are drawn
        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            
            .begin_render_pass(framebuffers.as_ref().unwrap()[image_num].clone(), false,
                               vec![[1.0, 1.0, 1.0, 1.0].into(),  1f32.into()])
            .unwrap()

            
            .draw_indexed(po_pipeline.clone(),
                  &dynamic_state,
                  ver_buf.clone(), index_buffer.clone(), set.clone(), ())
            .unwrap()


            .draw_indexed(pipeline_main.clone(),
                  &dynamic_state,
                  ver_buf.clone(), index_buffer_tri.clone(), set.clone(), ())
            .unwrap() 

            .end_render_pass()
            .unwrap()

            
            .build().unwrap();
        //present the swapchain to the  window
        let future = previous_frame_end.join(acquire_future)
            .then_execute(queue.clone(), command_buffer).unwrap()

           
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                previous_frame_end = Box::new(future) as Box<_>;
            }
            Err(vulkano::sync::FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Box::new(vulkano::sync::now(device.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Box::new(vulkano::sync::now(device.clone())) as Box<_>;
            }
        }


    
        let mut done = false;
        //event loop to capture user input, the function poll_events pops events from a stack and handles them
        //very slow when rendering for some reason.
        events_loop.poll_events(|ev| {
            match ev {
                winit::Event::WindowEvent { event, .. } => match event {
                    winit::WindowEvent::CloseRequested =>  done = true,
                    winit::WindowEvent::KeyboardInput {
                        input:
                            winit::KeyboardInput {
                                virtual_keycode: Some(virtual_code),
                                state: winit::ElementState::Released,
                                ..
                            },
                        ..
                    } => match virtual_code {
                        winit::VirtualKeyCode::Space => show = true,
                        winit::VirtualKeyCode::Left => { cam_x_pos -= 10.0;  },
                        winit::VirtualKeyCode::Right => { cam_x_pos += 10.0; },
                        _ => (),
                    },

                   _ => (),
                },
                
                _ => (),
            };
        });
         if done {
            return;
        } 
    }
}

//main function for gift wrap
fn gift_wrap_3(p: &Vec<Vertex>) -> Vec<Triangle> {
    let mut h = Vec::new();//for triangles
    let mut q = Vec::new();//stack for actual algorithm
    let size = p.len();
    let mut min_x = p[0].position[0];

    let mut p1 = Vertex { position: [0.0, 0.0, 0.0],
                          v_color: [0.0, 0.0, 0.0, 0.0],
                        index: 0,
                        };
    //grab rightmost element (least x value)
    for x in 0..size {
        if p[x].position[0] <= min_x {
            min_x = p[x].position[0];
            p1 = p[x].clone();
        }
    } 
   
    /*this code is similar to the make_triangle code. It uses the unit vector zeroed at x 
        coordinate (which gives a point that is one step below on the z and y axes). The plane angle maximized by this will give
        the coordinate with the greatest absolute value of their x-coordiante. This will maximize the plane angle, and because p1 is guaranteed
        to be included in the hull, this newly found point will be too
    */
    let mut candidate = -1 as i32;

    let p1_vec= cgmath::Vector3::new(p1.position[0],
                                    p1.position[1],
                                    p1.position[2]);

     //create temp p2 for edge by subtracting unit vector from p1   
    let p2 = p1_vec - cgmath::Vector3::new(0.0,
                                    1.0,
                                    1.0);

    let edge = p2 - p1_vec;
    for x in 0..size {
        
        //make sure the porspective index is not the point p1
        if x as u32 != p1.index {

            //create vector for math computation. This is tested point
            let p3 = cgmath::Vector3::new(p[x].position[0],
                                        p[x].position[1],
                                    p[x].position[2]);
            
            if candidate == -1 {
                candidate = x as i32;
            }
            //create point that currently give max angle of plane with edge.
            let pc = cgmath::Vector3::new(p[candidate as usize].position[0],
                                        p[candidate as usize].position[1],
                                    p[candidate as usize].position[2]);


            let norm_edge = cgmath::InnerSpace::normalize(edge);

            //distane vector from p3 to p1
            let v = p3 - p1_vec;

            //project v onto the edge p1-p2
            let dot = cgmath::dot(v, edge);

            //grab distance vector from v to its projection on edge. this is a vector parallel to plane create from p1-p2-p3
            let new_v = v - cgmath::Vector3::new(dot * edge.x, dot * edge.y, dot * edge.z);

            //distance vector from candidate vector and p1
            let c = pc - p1_vec;

            println!("{:?}", c);

            let dot = cgmath::dot(c, edge);

            //distance vector from c to its projection on edge
            let new_c = c - cgmath::Vector3::new(dot * edge.x, dot * edge.y, dot * edge.z);

            /*grab cross product of candidate plane vector and the tested plane vector
              if the candidate plane vector has a greater angle relative to the perpendicular to the edge, the 
              normal will point upwards, whereas if it is a lesser angle, then it will point downward
            */
            let mut n_1 = new_c.cross(new_v);

            //determine angle of normal relative to edge
            let result = cgmath::dot(n_1, edge);

            //if it is point upwards, make x the new candidate
            if result > 0.0 {
                candidate = x as i32;
            }
        }
    }

    let p2 = Vertex { position: [p[candidate as usize].position[0],
                                    p[candidate as usize].position[1],
                                   p[candidate as usize].position[2]],
                        v_color: [p[candidate as usize].v_color[0],
                                    p[candidate as usize].v_color[1],
                                   p[candidate as usize].v_color[2],
                                   p[candidate as usize].v_color[3]],
                        index: p[candidate as usize].index

                        };  
    //stack of edges for the algorithm to traverse
    let mut edges_to_see = Vec::new();
    //load first edge
    q.push(Edge {point1 : p2, point2: p1, processed: false});
    edges_to_see.push(Edge {point1 : p1, point2: p2, processed: false});
    while edges_to_see.len() > 0 {
        let curr_edge = edges_to_see.pop().unwrap();
        if processed(&q, curr_edge) == false {
            //println!("Hey: {:?} {:?}", curr_edge.point1, curr_edge.point2);
            let p3 = make_triangle(curr_edge, p, &h);

            if (p3.position[0] ==  0.0 as f32) && (p3.position[1] ==  0.0 as f32) && (p3.position[2] ==  0.0 as f32) {
                continue;
            } else {
                //add triangle to array 
                h.push(Triangle {point1: curr_edge.point1,
                                point2: curr_edge.point2,
                                point3: p3});
                //load edges to visit in counterclockwise rotation. this will effective BFS all over the hull
                q.push(Edge {point1 : curr_edge.point1, point2: curr_edge.point2, processed: false});  
                q.push(Edge {point1 : curr_edge.point2, point2: p3, processed: false});  
                q.push(Edge {point1 : p3, point2: curr_edge.point1, processed: false});
                 
                if processed(&q, Edge {point1 : curr_edge.point2, point2: curr_edge.point1, processed: false}) == false {
                    edges_to_see.push(Edge {point1 : curr_edge.point2, point2: curr_edge.point2, processed: false});
                }  
                if processed(&q, Edge {point1 : p3, point2: curr_edge.point2, processed: false}) == false {
                    edges_to_see.push(Edge {point1 : p3, point2: curr_edge.point2, processed: false});
                }  
                if processed(&q, Edge {point1 : curr_edge.point1, point2: p3, processed: false}) == false {
                    edges_to_see.push(Edge {point1 : curr_edge.point1, point2: p3, processed: false});
                }  
            }
        }
    }

    h
}


fn make_triangle(e: Edge, p: &Vec<Vertex>, t: &Vec<Triangle>) -> Vertex {
    let p_size = p.len();
    let t_size = t.len();
    let mut candidate = -1 as i32;
    for x in 0..p_size {
        let mut already = false;
        
        //check if point is one of edge points
        if p[x].index == e.point1.index || p[x].index == e.point2.index {

            continue;
        } else {
            //println!("Else {}", p[x].index);
            for i in 0 .. t_size {
                //check if triangle is already created
                if (t[i].point1.index == e.point1.index || t[i].point1.index == e.point2.index || t[i].point1.index == p[x].index) &&
                    (t[i].point2.index == e.point1.index || t[i].point2.index == e.point2.index || t[i].point2.index == p[x].index) &&
                    (t[i].point3.index == e.point1.index || t[i].point3.index == e.point2.index || t[i].point3.index == p[x].index) {
                        already = true;
                    }
            }
        }
        if already {
            continue;
        }


        if candidate == -1 {
            candidate = x as i32;
        }
        
        //create vectors using cgmath for computation

        let p1= cgmath::Vector3::new(e.point1.position[0],
                                    e.point1.position[1],
                                    e.point1.position[2]);

        
        let mut p2 = cgmath::Vector3::new(e.point2.position[0],
                                    e.point2.position[1],
                                    e.point2.position[2]);

        let edge = p2 - p1;

        

        let norm_edge = cgmath::InnerSpace::normalize(edge);
    
     //make sure the porspective index is not the point p1
        

            //create vector for math computation. This is the tested point
            let p3 = cgmath::Vector3::new(p[x].position[0],
                                        p[x].position[1],
                                    p[x].position[2]);
            
            if candidate == -1 {
                candidate = x as i32;
            }
            //create point that currently give max angle of plane with edge.
            let pc = cgmath::Vector3::new(p[candidate as usize].position[0],
                                        p[candidate as usize].position[1],
                                    p[candidate as usize].position[2]);


            let norm_edge = cgmath::InnerSpace::normalize(edge);

            //distane vector from p3 to p1
            let v = p3 - p1;

            //project v onto the edge p1-p2
            let dot = cgmath::dot(v, edge);

            //grab distance vector from v to its projection on edge. this is a vector parallel to plane create from p1-p2-p3
            let new_v = v - cgmath::Vector3::new(dot * edge.x, dot * edge.y, dot * edge.z);

            //distance vector from candidate vector and p1
            let c = pc - p1;


            let dot = cgmath::dot(c, edge);

            //distance vector from c to its projection on edge
            let new_c = c - cgmath::Vector3::new(dot * edge.x, dot * edge.y, dot * edge.z);

            /*grab cross product of candidate plane vector and the tested plane vector
              if the candidate plane vector has a greater angle relative to the perpendicular to the edge, the 
              normal will point upwards, whereas if it is a lesser angle, then it will point downward
            */
            let mut n_1 = new_c.cross(new_v);

            //determine angle of normal relative to edge
            let result = cgmath::dot(n_1, edge);

            //if it is point upwards, make x the new candidate
            if result > 0.0 {
                candidate = x as i32;
            }
        
    }
    return p[candidate as usize];
}
//this function simply checks if the algorithm has already process this edge and found maximum angle. 
fn processed(e_set: &Vec<Edge>, e: Edge) -> bool {
    let size = e_set.len();
    let mut res = false;
    if size > 0 {
        for i in 0 .. size {
            if e.point1.position[0] == e_set[i].point1.position[0] && e.point1.position[1] == e_set[i].point1.position[1]&& e.point1.position[2] == e_set[i].point1.position[2] 
            && e.point2.position[0] == e_set[i].point2.position[0] && e.point2.position[1] == e_set[i].point2.position[1]&& e.point2.position[2] == e_set[i].point2.position[2] {
                res = true;
            };
        }
    }
    return res;
}
// creates 15 random points of random colors
fn make_some_points() -> Vec<Vertex> {
    let mut v_set: Vec<Vertex> = Vec::new();
    let mut rng = rand::thread_rng();
    for i in 0 .. 15 {
        let mut x = rng.gen_range(-20.0, 20.0);
        let mut y = rng.gen_range(-20.0, 20.0);
        let mut z = rng.gen_range(-20.0, 20.0);
        let size = v_set.len();
        for j in 0 .. size {
            if v_set[j].position[0] == x {
                x = rng.gen_range(-20.0, 20.0);
            }
            if v_set[j].position[1] == y {
                y = rng.gen_range(-20.0, 20.0);
            }
            if v_set[j].position[2] == z {
                z = rng.gen_range(-20.0, 20.0);
            } 
        }
        let v = Vertex { position: [x, y, z],
                        v_color: [rng.gen_range(0.0, 1.0),
                        rng.gen_range(0.0, 1.0),
                        rng.gen_range(0.0, 1.0), 1.0],
                    index: i,
        };
        v_set.push(v);
    }
    v_set
} 
