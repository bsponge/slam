#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <limits>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <SFML/Window.hpp>
#include <SFML/System/Time.hpp>

const char* vertexSource =
"#version 150 core\n"
"in vec3 position;\n"
"uniform mat4 model;\n"
"uniform mat4 view;\n"
"uniform mat4 proj;\n"
" "
"void main() {\n"
" gl_Position = proj * view * model * vec4(position, 1.0);\n"
"}\n";

const char* fragmentSource =
"#version 150 core\n"
"out vec4 Color;\n"
"uniform vec3 color;\n"
" "
"void main() {\n"
" Color = vec4(color, 1.0);\n"
"}\n";

bool firstMouse = true;
int lastX, lastY;
double yaw = 45.0;
double pitch = 0.0;

glm::vec3 cameraPos = glm::vec3(0.3f, 0.3f, 0.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

void setCamera(GLint uniView, sf::Int64 time, sf::Window& window) {
  sf::Vector2i localPosition = sf::Mouse::getPosition(window);
  sf::Vector2i position;
  bool relocation = false;

  if (localPosition.x <= 0) {
    position.x = window.getSize().x - 1;
    position.y = localPosition.y;
    relocation = true;
  }
  if (localPosition.x >= window.getSize().x - 1) {
    position.x = 0;
    position.y = localPosition.y;
    relocation = true;
  }
  if (localPosition.y <= 0) {
    position.y = window.getSize().y;
    position.x = localPosition.x;
    relocation = true;
  }
  if (localPosition.y >= window.getSize().y - 1) {
    position.y = 0;
    position.x = localPosition.x;
    relocation = true;
  }
  if (relocation) {
    sf::Mouse::setPosition(position, window);
    firstMouse = true;
    localPosition = sf::Mouse::getPosition(window);
  }
  
  
  if (firstMouse) {
    lastX = localPosition.x;
    lastY = localPosition.y;
    firstMouse = false;
  }

  double xoffset = localPosition.x - lastX;
  double yoffset = localPosition.y - lastY;
  lastX = localPosition.x;
  lastY = localPosition.y;

  double sensitivity = 0.001;
  double cameraSpeed = 0.003 * time;
  
  xoffset *= sensitivity;
  yoffset *= sensitivity;

  yaw += xoffset * cameraSpeed;
  pitch -= yoffset * cameraSpeed;

  if (pitch > 89.0) {
    pitch = 89.0;
  }
  if (pitch < -89.0) {
    pitch = -89.0;
  }

  glm::vec3 front;
  front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
  front.y = sin(glm::radians(pitch));
  front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
  cameraFront = glm::normalize(front);

  glm::mat4 view;
  view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
  glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));
}

int loadPointsFromFrame(int frame, int buffer, const char* points_filename, const char* points_num_filename) {
  std::vector<float> vec;
  std::ifstream points_file;
  std::ifstream points_num_file;
  points_file.open(points_filename);
  points_num_file.open(points_num_filename);
  int num_of_points_to_read = 0;
  int real_size = 0;
  if (points_file.is_open() && points_num_file.is_open()) {
    int i = 0;
    while (!points_num_file.eof() && i < frame) {
      int  tmp = 0;
      points_num_file >> tmp;
      num_of_points_to_read += tmp*3;
      ++i;
    }
    float* tmp_arr = new float[num_of_points_to_read];
    float max_x = std::numeric_limits<float>::min();
    float max_y = max_x;
    float max_z = max_x;
    for (i = 0; !points_file.eof() && i < num_of_points_to_read; i += 3) {
      for (int j = 0; !points_file.eof() && j < 3; ++j) {
        points_file >> tmp_arr[i+j];
        ++real_size;
      } 
      if (abs(tmp_arr[i] > max_x) && i < real_size) {
        max_x = abs(tmp_arr[i]);
      }
      if (abs(tmp_arr[i+1]) > max_y && i < real_size+1) {
        max_y = abs(tmp_arr[i+1]);
      }
      if (abs(tmp_arr[i+2]) > max_z && i < real_size+2) {
        max_z = abs(tmp_arr[i+2]);
      }
    }

    for (i = 0; i < real_size; i += 3) {
      tmp_arr[i] /= 1000;
      tmp_arr[i+1] /= 1000;
      tmp_arr[i+2] /= 1000;
    }

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GL_FLOAT)*real_size, tmp_arr, GL_STATIC_DRAW);

    delete [] tmp_arr;

    points_file.close();
    points_num_file.close();
  } else {
    std::cout << "points_file: " << points_filename << " is " << points_file.is_open() << std::endl;
    std::cout << "points_num_file: " << points_num_filename << " is " << points_num_file.is_open() << std::endl;
  }

  return real_size;
}


int loadVertices(int buffer, const char* filename) {
  std::vector<float> vec;
  std::ifstream ifstream;
  ifstream.open(filename);
  float output;
  if (ifstream.is_open()) {
    while (!ifstream.eof()) {
      ifstream >> output;
      vec.push_back(output);
    }
  } else {
    std::cout << "File " << filename << " is closed" << std::endl;
  }
  
  float* arr = new float[vec.size()];
  float max_x = std::numeric_limits<float>::min();
  float max_y = max_x;
  float max_z = max_x;
  for (int i = 0; i < vec.size(); i += 3) {
    if (abs(vec[i] > max_x)) {
      max_x = abs(vec[i]);
    }
    if (abs(vec[i+1]) > max_y) {
      max_y = abs(vec[i+1]);
    }
    if (abs(vec[i+2]) > max_z) {
      max_z = abs(vec[i+2]);
    }
    arr[i] = vec[i];
    arr[i+1] = vec[i+1];
    arr[i+2] = vec[i+2];
  } 

  for (int i = 0; i < vec.size(); i += 3) {
    arr[i] /= max_x;
    arr[i+1] /= max_y;
    arr[i+2] /= max_z;
  }

  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(GL_FLOAT)*vec.size(), arr, GL_STATIC_DRAW); 
  int size = vec.size();

  delete [] arr;
  return size;
}



int main() {


  int frames_num = 10;

  sf::ContextSettings settings;
  settings.depthBits = 24;
  settings.stencilBits = 8;

  sf::Window window = sf::Window(sf::VideoMode(1000, 800, 32), "SLAM", sf::Style::Titlebar | sf::Style::Close, settings);
  glewExperimental = GL_TRUE;
  glewInit();
  //window.setMouseCursorGrabbed(true);
  //window.setMouseCursorVisible(false);
  
  window.setFramerateLimit(60);

  GLuint vao;
  glGenBuffers(1, &vao);


  GLuint vbo;
  glGenBuffers(1, &vbo);

  float* points = nullptr;

  const char* points_filename = "../../camera_pose";
  const char* points_num_filename = "../../points_in_frame";
  //int points_size = loadVertices(vbo, "/home/js/python/slam/points");
  int points_size = loadPointsFromFrame(frames_num, vbo, points_filename, points_num_filename);
  if (points_size == 0) {
    std::cout << "ERROR WHILE LOADING VERTICES" << std::endl;
    return 1;
  }


  std::cout << "points size: " << points_size << std::endl;

  char infoLog[512];
  
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexSource, NULL);
  glCompileShader(vertexShader);

  GLint status;
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &status);
  if (!status) {
    glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
    std::cout << "ERROR" << status << std::endl;
    std::cout << "ERROR::SHADER::VERTEX::COMPILATION::FAILED\n" << infoLog << std::endl;
  } else {
    std::cout << "Compilation vertexShader OK" << std::endl;
  }


  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
  glCompileShader(fragmentShader);

  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &status);
  if (!status) {
    glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
    std::cout << "ERROR" << status << std::endl;
    std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION::FAILED\n" << infoLog << std::endl;
  } else {
    std::cout << "Compilation fragmentShader OK" << std::endl;
  }

  GLuint shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);

  glBindVertexArray(vao);

  GLint posAttrib = glGetAttribLocation(shaderProgram, "position"); 
  glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GL_FLOAT), (void*)0);
  glEnableVertexAttribArray(posAttrib);

  glUseProgram(shaderProgram);

  glm::mat4 model = glm::mat4(1.0f);
  model = glm::rotate(model, glm::radians(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

  GLint uniTrans = glGetUniformLocation(shaderProgram, "model");
  glUniformMatrix4fv(uniTrans, 1, GL_FALSE, glm::value_ptr(model));

  glm::mat4 view;


  view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

  GLint uniView = glGetUniformLocation(shaderProgram, "view");
  glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));

  glm::mat4 proj = glm::perspective(glm::radians(45.0f), 800.0f / 800.0f, 0.06f, 100.0f);
  GLint uniProj = glGetUniformLocation(shaderProgram, "proj");

  glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));

  glm::vec3 color(1.0, 1.0, 1.0);
  GLint unicolor = glGetUniformLocation(shaderProgram, "color");
  glUniform3fv(unicolor, 1, glm::value_ptr(color));

  sf::Clock clock;
  sf::Time time;


  // Rozpoczęcie pętli zdarzeń

  bool running = true;

  glEnable(GL_DEPTH_TEST);
  int licznik = 0;

  bool mouseMoving = false;

  


  while (running) {
    time = clock.restart();
    licznik++;
    float cameraSpeed = 0.000005f * time.asMicroseconds();
    float ffps = 1000000 / time.asMicroseconds();
    if (licznik > ffps) {
      window.setTitle(std::to_string(ffps));
      licznik = 0;
    }
    sf::Event windowEvent;

    while (window.pollEvent(windowEvent)) {
      switch (windowEvent.type) {
        case sf::Event::Closed:
          running = false;
          break;
        //case sf::Event::MouseMoved:
        case sf::Event::MouseMoved:
          if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
            setCamera(uniView, time.asMicroseconds(), window);
          break;
        case sf::Event::KeyPressed:
          switch (windowEvent.key.code) {
            case sf::Keyboard::Escape:
              running = false;
              break;
            case sf::Keyboard::W:
              cameraPos += cameraSpeed * cameraFront;
              view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
              uniView = glGetUniformLocation(shaderProgram, "view");
              glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));
              break;
            case sf::Keyboard::A:
              cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
              view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
              uniView = glGetUniformLocation(shaderProgram, "view");
              glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));
              break;
            case sf::Keyboard::S:
              cameraPos -= cameraSpeed * cameraFront;
              view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
              uniView = glGetUniformLocation(shaderProgram, "view");
              glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));
              break;
            case sf::Keyboard::D:
              cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
              view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
              uniView = glGetUniformLocation(shaderProgram, "view");
              glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));
              break;
            case sf::Keyboard::Space:
              cameraPos.y += cameraSpeed;
              view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
              uniView = glGetUniformLocation(shaderProgram, "view");
              glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));
              break;
            case sf::Keyboard::Z:
              cameraPos.y -= cameraSpeed;
              view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
              uniView = glGetUniformLocation(shaderProgram, "view");
              glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));
              break; 
            case sf::Keyboard::R:
              ++frames_num;
              points_size = loadPointsFromFrame(frames_num, vbo, points_filename, points_num_filename);
              std::cout << "points_size: " << points_size << std::endl;
              break;
            case sf::Keyboard::E:
              if (frames_num > 1) {
                --frames_num;
                points_size = loadPointsFromFrame(frames_num, vbo, points_filename, points_num_filename);
              }
              break;
          }
          break;
      }
    } 
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    color.x += 0.1;
    color.x = sin(color.x);
    glUniform3fv(unicolor, 1, glm::value_ptr(color));

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glPointSize(1.0);
    glDrawArrays(GL_POINTS, 0, points_size/3);

    window.display();
  }

  glDeleteProgram(shaderProgram);
  glDeleteShader(fragmentShader);
  glDeleteShader(vertexShader);
  glDeleteBuffers(1, &vbo);
  glDeleteVertexArrays(1, &vao);

  return 0;
}
