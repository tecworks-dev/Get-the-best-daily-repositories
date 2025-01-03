#include "renderer.h"
#include <numeric>


static const char* vtxShader =
R"sv(#version 310 es

uniform vec4 vcolor[7];

in vec4 position;
//in vec4 vtxcolor;

out vec4 color;

void main() {
    gl_Position = position;
    //color = vcolor[gl_InstanceID % 7];
    color = vcolor[gl_VertexID % 7];
}
)sv";

static const char* fragShader =
R"sf(#version 310 es
precision mediump float;
precision highp int;


in vec4 color;
layout(location=0) out vec4 FragColor;

layout (binding = 0, offset = 0) uniform atomic_uint counter;
layout (std140) uniform UniformBlock {
    vec4 uniformcolor;
};

uniform uint frags_to_shade;

void main() {
    uint frags_shaded = atomicCounterIncrement(counter);
    if (frags_shaded > frags_to_shade) {
        FragColor = vec4(0.);
    }
    else {
        FragColor = color;
    }
}
)sf";

Renderer::Renderer() : num_vertices(0), num_indices(0), ibo(0), vbo(0), acbo(0), program(0),
                        gen(rd())
                    {
                        int num = 1;
                        float stride = 2.f / (float)num;
                        for (int i = 0; i < num; ++i) {
                            float start = -1.f + stride * (float) i;
                            distributions.emplace_back(start + stride / 2.f,
                                                       start + stride / 2. + 1.f);
                        }
                        nudge_distribution = std::uniform_real_distribution<float>(0., stride);

                        distribution = std::uniform_real_distribution<float>(-1, 1);
                    }

Renderer::~Renderer()
{
    if (program)
    {
        glDeleteProgram(program);
    }
    if (ibo)
    {
        glDeleteBuffers(1, &ibo);
    }
    if (vbo)
    {
        glDeleteBuffers(1, &vbo);
    }
}

static const char* errorToStr(GLenum error)
{
    switch(error)
    {
        case GL_INVALID_ENUM:
            return "GL_INVALID_ENUM";
        case GL_INVALID_OPERATION:
            return "GL_INVALID_OPERATION";
        case GL_INVALID_VALUE:
            return "GL_INVALID_VALUE";
#ifdef GL_INVALID_INDEX
        case GL_INVALID_INDEX:
            return "GL_INVALID_INDEX";
#endif // GL_INVALID_INDEX
        default:
            return "Unknown error";
    }
}

bool Renderer::init()
{
#ifdef GL_PROFILE_GL3
    if (vao == 0)
    {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
    }
#endif
    glGenBuffers(1, &vbo);

    glCheckError();

    glGenBuffers(1, &ibo);

    glGenBuffers(1, &acbo);
    GLuint init_count = 0;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, acbo);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), &init_count, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, acbo);
    glCheckError();

    glEnable(GL_BLEND);
    glCheckError();
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glCheckError();

    if (!compileShaders())
    {
        Log(LOG_ERROR) << "Initialization failed";
        return false;
    }
    glCheckError();
    glUseProgram(program);
    glCheckError();

    glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);

    ubo_data.color = glm::vec4(0.5, 0.5, 0, 1.);
    ubo_data.frag_count = frag_count;
    glBufferData(GL_UNIFORM_BUFFER, sizeof(ubo_data), &ubo_data, GL_DYNAMIC_DRAW);
    glCheckError();
    GLuint idx = glGetUniformBlockIndex(program, "UniformBlock");
    glCheckError();
    GLuint ubb = 1;
    glUniformBlockBinding(program, idx, ubb);
    glBindBufferBase(GL_UNIFORM_BUFFER, ubb, ubo);
    glCheckError();

    GLint loc = glGetUniformLocation(program, "frags_shaded");
    glCheckError();
    glUniform1ui(loc, ubo_data.frag_count);
    glCheckError();

    glGenBuffers(1, &vubo);
    glBindBuffer(GL_UNIFORM_BUFFER, vubo);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(vubo_t), &vubo_data, GL_DYNAMIC_DRAW);
    glCheckError();
    GLint vloc = glGetUniformLocation(program, "vcolor");
    glCheckError();
    glUniform4fv(vloc, 7, &vubo_data.vcolor[0].x);

    return true;
}

void Renderer::draw()
{
#ifdef GL_PROFILE_GL3
    glBindVertexArray(vao);
#endif

    glEnable(GL_DEPTH_TEST);

    glCheckError();
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glCheckError();

    size_t vtxcount = triangle_count * 3;
    if (vtxcount != ibData.size()) {
        ibData.clear();
        for (size_t i = 0; i < vtxcount; ++i) {
            ibData.emplace_back(ibDb[i % ibDb.size()]);
        }
    }
    glBufferData(GL_ARRAY_BUFFER, vbData.size() * sizeof(vtxData), vbData.data(),
                 GL_STATIC_DRAW);
    glCheckError();
    glEnableVertexAttribArray(0);
    glCheckError();
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(vtxData), (void*)0);
    glCheckError();

    //    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(vtxData), (void*)offsetof(vtxData, color));
    //    glEnableVertexAttribArray(1);
    //glCheckError();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

    glCheckError();

        glBufferData(GL_ELEMENT_ARRAY_BUFFER, ibData.size() * sizeof(uint16_t), ibData.data(),
                     GL_STATIC_DRAW);
        glCheckError();

    GLuint init_count = 0;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, acbo);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), &init_count, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, acbo);
    glCheckError();

    glBindBuffer(GL_UNIFORM_BUFFER, ubo);

    ubo_data.color = glm::vec4(0.5, 0.5, 0, 1.);
    ubo_data.frag_count = frag_count;
    glBufferData(GL_UNIFORM_BUFFER, sizeof(ubo_data), &ubo_data, GL_DYNAMIC_DRAW);
    glCheckError();
    GLuint idx = glGetUniformBlockIndex(program, "UniformBlock");
    glCheckError();

    GLuint ubb = 1;
    glUniformBlockBinding(program, idx, ubb);
    glBindBufferBase(GL_UNIFORM_BUFFER, ubb, ubo);
    glCheckError();

    GLint loc = glGetUniformLocation(program, "frags_to_shade");
    glCheckError();
    glUniform1ui(loc, ubo_data.frag_count);
    glCheckError();

    glDrawElements(GL_TRIANGLES, ibData.size(), GL_UNSIGNED_SHORT, 0);
    glCheckError();
}

static GLint compileShader(GLenum shaderType, const char* shaderSrc)
{
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &shaderSrc, NULL);
    glCompileShader(shader);
    int status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE)
    {
        // Assume we only care about vertex and fragment shaders
        Log(LOG_ERROR) << "Could not compile shader! Shader type: " << ((shaderType == GL_VERTEX_SHADER) ? "vertex" : "fragment");
        GLint logLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
        std::vector<char> infoLog(logLength + 1);
        glGetShaderInfoLog(shader, infoLog.size(), &logLength, infoLog.data());
        Log(LOG_ERROR) << "Error log: " << infoLog.data();
        Log(LOG_ERROR) << "Shader source: " << shaderSrc;
        glDeleteShader(shader);
        return -1;
    }
    return shader;
}

bool Renderer::compileShaders() {
    GLint vertexShader = compileShader(GL_VERTEX_SHADER, vtxShader);
    GLint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragShader);
    if (vertexShader < 0 || fragmentShader < 0)
    {
        // Delete any shaders that were actually compiled
        if (vertexShader >= 0) {glDeleteShader(vertexShader);}
        if (fragmentShader >= 0) {glDeleteShader(fragmentShader);}
        return false;
    }

    program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status != GL_TRUE)
    {
        Log(LOG_ERROR) << "Could not link shaders; interface mismatch?";
        GLint logLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
        std::vector<char> infoLog(logLength + 1);
        glGetProgramInfoLog(program, infoLog.size(), &logLength, infoLog.data());
        Log(LOG_ERROR) << "Error log: " << infoLog.data();
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        glDeleteProgram(program);
        return false;
    }

    return true;
}

void Renderer::drawRandomTris(int tricount) {
//    auto start = std::chrono::steady_clock::now();
    int vtxcount = tricount * 3;
    if (vbDataRandom.size() != vtxcount) {
//        if (vbDataRandom.size() > vtxcount) {
//            vbDataRandom.resize(vtxcount);
//        }
        /*else {*/ // size() < vtxcount
//            std::uniform_real_distribution<float> distribution(-0.1, 0.0);
        vbDataRandom.clear();
            int vtxcount2add = vtxcount - vbDataRandom.size();
            int last_triidx = -1;
            float nudge_x = 0.f;
            float nudge_y = 0.f;
        for (int i = 0; i < vtxcount2add; ++i) {
                int tricount = vtxcount2add / 3;
                int triidx = i / 3;
                float z = (float)(tricount - 1 - triidx) / (float)tricount * 2. - 1.;
//                size_t sq_num = distributions.size() * distributions.size();
//                size_t idx = triidx % sq_num;
//                size_t tile_x = idx % distributions.size();
//                size_t tile_y = idx / distributions.size();
//                auto& distribution_x = distributions[tile_x];
//                auto& distribution_y = distributions[tile_y];
//                if (last_triidx != triidx) {
//                    last_triidx = triidx;
//                    nudge_x = nudge_distribution(gen);
//                    nudge_y = nudge_distribution(gen);
//                }



//                vtxData v {
//                    .pos = glm::vec4(distribution_x(gen) + nudge_x, distribution_y(gen) + nudge_y, z, 1.)
//                };
                vtxData v {
                        .pos = glm::vec4(distribution(gen), distribution(gen), z, 1.)
                };

                //v.pos.y = 1. - v.pos.x - v.pos.z - v.pos.w;
                vbDataRandom.push_back(v);
            }
//        }
    }
//    auto end = std::chrono::steady_clock::now();
//    uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    glBufferData(GL_ARRAY_BUFFER, vbDataRandom.size() * sizeof(vtxData), vbDataRandom.data(),
        GL_STATIC_DRAW);
    glCheckError();
    glDrawArrays(GL_TRIANGLES, 0, vtxcount);
    glCheckError();
//    return ms;
}
