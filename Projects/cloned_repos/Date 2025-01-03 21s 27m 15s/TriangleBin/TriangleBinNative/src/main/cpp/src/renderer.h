#ifndef RENDERER_H
#define RENDERER_H

#ifdef GL_PROFILE_GL3
#include "GL/glew.h"
#else
#include <GLES3/gl32.h>
#endif

#include "logger.h"

#include <vector>
#include <random>
#include <glm/common.hpp>
#include <glm/matrix.hpp>

class Renderer {
public:
    Renderer();
    ~Renderer();
    bool init();
    void draw();
    void drawRandomTris(int count);
    int triangle_count = 2;
    uint32_t frag_count = 0;

private:
#ifdef GL_PROFILE_GL3
    static GLuint g_vao;
#endif
    int num_vertices;
    int num_indices;
    GLuint ibo;
    GLuint vbo;
    GLuint ubo;
    GLuint vubo;
    GLuint acbo;
#ifdef GL_PROFILE_GL3
    GLuint vao = 0;
#endif

    GLuint program;
    struct vtxData {
        glm::vec4 pos;
        glm::vec4 color;
    };

    std::vector<vtxData> vbData {
            { glm::vec4(-1,  1, 0,  1) },
            { glm::vec4( 1,  1, 0,  1) },
            { glm::vec4(-1, -1, 0,  1) },
//          { glm::vec4( 1,  1, 0,  1) },
//          { glm::vec4(-1, -1, 0,  1) },
            { glm::vec4( 1, -1, 0,  1) }
    };

    std::vector<uint16_t> ibDb { 0, 1, 2, 1, 2, 3 };
    std::vector<uint16_t> ibData { 0, 1, 2, 1, 2, 3 };

    struct ubo_t {
        glm::vec4 color = glm::vec4(1., 0., 0., 1.);
        GLuint frag_count = 0;
    };

    struct vubo_t {
        glm::vec4 vcolor[7];
    };

    vubo_t vubo_data = {
            .vcolor = {
                glm::vec4(1, 0, 0, 1),
                glm::vec4(0, 1, 0, 1),
                glm::vec4(0, 0, 1, 1),
                glm::vec4(1, 1, 0, 1),
                glm::vec4(1, 0, 1, 1),
                glm::vec4(0, 1, 1, 1),
                glm::vec4(1, 1, 1, 1)
            }
    };

    struct ssbo_t {
        int tri_counter = 0;
    };

    ubo_t ubo_data;
    ssbo_t ssbo_data;

    std::vector<vtxData> vbDataRandom;
    std::random_device rd;
    std::mt19937 gen;

    std::vector<std::uniform_real_distribution<float>> distributions;
    std::uniform_real_distribution<float> nudge_distribution;
    std::uniform_real_distribution<float> distribution;

    bool compileShaders();

};

#define glCheckError() {GLenum error = glGetError(); if (error != GL_NO_ERROR) {Log(LOG_ERROR) << "GL error at " << __FILE__ << "@" << __LINE__ << ": " << error << " (" << errorToStr(error) << ")";}}



#endif //RENDERER_H
