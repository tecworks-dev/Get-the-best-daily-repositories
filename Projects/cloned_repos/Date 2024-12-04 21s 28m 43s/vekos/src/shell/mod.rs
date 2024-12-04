/*
 * Copyright 2023-2024 Juan Miguel Giraldo
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use alloc::string::String;
use crate::process::ProcessState;
use alloc::vec::Vec;
use crate::ALLOCATOR;
use alloc::collections::BTreeMap;
use crate::vga_buffer::{WRITER, Color};
use crate::serial_println;
use alloc::format;
use crate::fs::{FILESYSTEM};
use crate::process::PROCESS_LIST;
use crate::syscall::KEYBOARD_BUFFER;
use crate::fs::FileSystem;
use core::fmt::Write;
use crate::Process;
use crate::fs::FsError;
mod commands;
use crate::vga_buffer::BUFFER_WIDTH;
use crate::fs::validate_path;
use crate::MEMORY_MANAGER;
use crate::fs::normalize_path;
use crate::vga_buffer::ColorCode;
use crate::alloc::string::ToString;
use crate::vga_buffer::BUFFER_HEIGHT;

mod display;
use display::ShellDisplay;
mod parser;
mod executor;
use executor::CommandExecutor;
use parser::{Parser, TokenType};

#[derive(Debug)]
pub enum ShellError {
    CommandNotFound,
    InvalidArguments,
    ExecutionFailed,
    IOError,
    PermissionDenied,
    PathNotFound,
    InvalidPath,
    EnvironmentError,
    InternalError,
    BufferOverflow,
    SyntaxError,
    NotADirectory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitCode {
    Success = 0,
    Failure = 1,
    CommandNotFound = 127,
    InvalidArgument = 2,
}

impl ExitCode {
    pub fn from_i32(code: i32) -> Self {
        match code {
            0 => ExitCode::Success,
            1 => ExitCode::Failure,
            2 => ExitCode::InvalidArgument,
            127 => ExitCode::CommandNotFound,
            _ => ExitCode::Failure,
        }
    }
}

pub type ShellResult = Result<ExitCode, ShellError>;

pub struct Shell {
    input_buffer: InputBuffer,
    display: ShellDisplay,
    current_dir: String,
    is_running: bool,
    history: Vec<String>,
    history_position: usize,
    env_vars: BTreeMap<String, String>,
    prompt_color: Color,
    text_color: Color,
    executor: CommandExecutor,
}

#[derive(Debug)]
struct InputBuffer {
    buffer: Vec<u8>,
    cursor_position: usize,
    render_offset: usize,
}


impl InputBuffer {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(1024),
            cursor_position: 0,
            render_offset: 0,
        }
    }

    fn insert(&mut self, byte: u8) -> bool {
        if self.buffer.len() >= 1024 {
            return false;
        }
        
        self.buffer.insert(self.cursor_position, byte);
        self.cursor_position += 1;
        
        
        if self.cursor_position > self.render_offset + BUFFER_WIDTH {
            self.render_offset = self.cursor_position - BUFFER_WIDTH;
        }
        true
    }

    fn backspace(&mut self) -> bool {
        if self.cursor_position > 0 {
            self.cursor_position -= 1;
            self.buffer.remove(self.cursor_position);
            true
        } else {
            false
        }
    }

    fn delete(&mut self) -> bool {
        if self.cursor_position < self.buffer.len() {
            self.buffer.remove(self.cursor_position);
            true
        } else {
            false
        }
    }

    fn move_cursor_left(&mut self) -> bool {
        if self.cursor_position > 0 {
            self.cursor_position -= 1;
            true
        } else {
            false
        }
    }

    fn move_cursor_right(&mut self) -> bool {
        if self.cursor_position < self.buffer.len() {
            self.cursor_position += 1;
            true
        } else {
            false
        }
    }

    fn clear(&mut self) {
        self.buffer.clear();
        self.cursor_position = 0;
        self.render_offset = 0;
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn iter(&self) -> core::slice::Iter<'_, u8> {
        self.buffer.iter()
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }
}

impl Shell {
    pub fn new() -> Result<Self, ShellError> {
        Ok(Self {
            input_buffer: InputBuffer::new(),
            display: ShellDisplay::new(),
            current_dir: String::from("/"),
            is_running: true,
            history: Vec::new(),
            history_position: 0,
            env_vars: BTreeMap::new(),
            prompt_color: Color::Green,
            text_color: Color::White,
            executor: CommandExecutor::new(),
        })
    }

    pub fn init(&mut self) -> ShellResult {
        {
            let mut writer = WRITER.lock();
            writer.clear_screen();
            writer.enable_cursor();
            writer.write_str("VEKOS Shell v0.0.1\n").map_err(|_| ShellError::IOError)?;
            writer.write_str("Type 'help' for available commands\n\n").map_err(|_| ShellError::IOError)?;
        }

        let mut process_list = PROCESS_LIST.lock();
        let mut memory_manager = MEMORY_MANAGER.lock();
        
        if process_list.current().is_none() {
            if let Some(mm) = memory_manager.as_mut() {
                match Process::new(mm) {
                    Ok(mut init_process) => {
                        init_process.current_dir = String::from("/");
                        process_list.add(init_process)
                            .map_err(|_| ShellError::InternalError)?;
                    },
                    Err(_) => return Err(ShellError::InternalError),
                }
            }
        }
    
        {
            let mut keyboard_buffer = KEYBOARD_BUFFER.lock();
            keyboard_buffer.clear();
        }
    
        serial_println!("Shell initialized successfully");
        Ok(ExitCode::Success)
    }

    pub fn general_update_current_dir(&mut self, new_path: String) -> Result<(), ShellError> {
        let normalized = normalize_path(&new_path);
        self.current_dir = normalized;
        Ok(())
    }

    pub fn run(&mut self) -> ShellResult {
        while self.is_running {
            self.display_prompt()?;
            
            match self.read_command()? {
                ExitCode::Success => continue,
                exit_code => return Ok(exit_code),
            }
        }
    
        Ok(ExitCode::Success)
    }

    fn handle_pwd(&mut self) -> Result<(), &'static str> {
        use x86_64::instructions::interrupts;
        interrupts::without_interrupts(|| {
            let mut writer = WRITER.lock();
            writer.write_byte(b'/');
            writer.write_byte(b'\n');
        });
        Ok(())
    }

    fn display_prompt(&mut self) -> ShellResult {
        self.display.set_prompt(format!("{}> ", self.current_dir));
        let prompt_pos = self.display.render_prompt();
        
        
        let mut writer = WRITER.lock();
        writer.set_cursor_position(prompt_pos, BUFFER_HEIGHT - 1);
        
        Ok(ExitCode::Success)
    }

    fn read_command(&mut self) -> ShellResult {
        let mut command_complete = false;
        let initial_column = self.display.get_cursor_position();
        let prompt_len = self.display.get_prompt().len();
        
        while !command_complete {
            let mut keyboard_buffer = KEYBOARD_BUFFER.lock();
            
            if let Some(&byte) = keyboard_buffer.last() {
                serial_println!("Shell received byte: {} ({})", byte as char, byte);
        
                keyboard_buffer.pop();
                drop(keyboard_buffer);
                
                match byte {
                    b'\n' => {
                        
                        {
                            let mut writer = WRITER.lock();
                            writer.write_byte(b'\n');
                        }
                        
                        
                        let input = String::from_utf8_lossy(&self.input_buffer.buffer).into_owned();
                        let trimmed = input.trim();
                        
                        
                        self.input_buffer.clear();
                        self.display.clear_line();
                        
                        if !trimmed.is_empty() {
                            let mut parser = Parser::new(trimmed);
                            match parser.parse() {
                                Ok(tokens) => {
                                    if !tokens.is_empty() {
                                        
                                        if self.history.len() >= 1000 {
                                            self.history.remove(0);
                                        }
                                        self.history.push(trimmed.to_string());
                                        self.history_position = self.history.len();
                                        
                                        
                                        let command = tokens[0].value.clone();
                                        let args = tokens[1..]
                                            .iter()
                                            .filter(|t| t.token_type == TokenType::Argument)
                                            .map(|t| t.value.clone())
                                            .collect::<Vec<_>>();
                                        
                                        let result = self.execute_command(&command, &args);

                                        if result.is_ok() {
                                            let mut writer = WRITER.lock();
                                            if writer.column_position > 0 {
                                                writer.write_byte(b'\n');
                                            }
                                        }
                                    }
                                }
                                Err(_) => {
                                    self.display.display_error(&ShellError::SyntaxError);
                                    let mut writer = WRITER.lock();
                                    writer.write_byte(b'\n');
                                }
                            }
                        }
                        
                        command_complete = true;
                        continue;
                    },
                                        
                    
                    27 => {
                        let mut keyboard_buffer = KEYBOARD_BUFFER.lock();
                        if keyboard_buffer.len() >= 2 {
                            let seq = (keyboard_buffer[0], keyboard_buffer[1]);
                            keyboard_buffer.drain(0..2);
                            drop(keyboard_buffer);
                            
                            match seq {
                                
                                (91, 68) => {
                                    if self.input_buffer.move_cursor_left() {
                                        self.display.move_cursor(
                                            initial_column + self.input_buffer.cursor_position
                                        );
                                    }
                                },
                                
                                (91, 67) => {
                                    if self.input_buffer.move_cursor_right() {
                                        self.display.move_cursor(
                                            initial_column + self.input_buffer.cursor_position
                                        );
                                    }
                                },
                                
                                (91, 65) => {
                                    if self.history_position > 0 {
                                        self.history_position -= 1;
                                        if let Some(previous_cmd) = self.history.get(self.history_position) {
                                            
                                            self.input_buffer.clear();
                                            
                                            
                                            self.input_buffer.buffer.extend(previous_cmd.bytes());
                                            self.input_buffer.cursor_position = previous_cmd.len();
                                            
                                            
                                            self.display.redraw_line(
                                                &self.input_buffer.buffer,
                                                self.input_buffer.cursor_position
                                            );
                                        }
                                    }
                                },
                                
                                (91, 66) => {
                                    if self.history_position < self.history.len() {
                                        self.history_position += 1;
                                        let cmd = if self.history_position == self.history.len() {
                                            ""
                                        } else {
                                            &self.history[self.history_position]
                                        };
                                        
                                        
                                        self.input_buffer.clear();
                                        
                                        
                                        self.input_buffer.buffer.extend(cmd.bytes());
                                        self.input_buffer.cursor_position = cmd.len();
                                        
                                        
                                        self.display.redraw_line(
                                            &self.input_buffer.buffer,
                                            self.input_buffer.cursor_position
                                        );
                                    }
                                },
                                _ => {}
                            }
                        } else {
                            drop(keyboard_buffer);
                        }
                    },
                    
                    
                    8 | 127 => {
                        if self.input_buffer.backspace() {
                            
                            self.display.redraw_line(
                                &self.input_buffer.buffer,
                                self.input_buffer.cursor_position
                            );
                        }
                    },
                    
                    
                    32..=126 => {
                        
                        let available_space = BUFFER_WIDTH.saturating_sub(2) 
                                                        .saturating_sub(initial_column);
                                                        
                        if self.input_buffer.buffer.len() < available_space {
                            
                            if self.input_buffer.insert(byte) {
                                
                                self.display.redraw_line(
                                    &self.input_buffer.buffer,
                                    self.input_buffer.cursor_position
                                );
                                
                                
                                let new_cursor_pos = initial_column + self.input_buffer.cursor_position;
                                if new_cursor_pos < BUFFER_WIDTH {
                                    WRITER.lock().set_cursor_position(new_cursor_pos, BUFFER_HEIGHT - 1);
                                }
                            }
                        }
                    },
                    
                    _ => {} 
                }

                self.display.redraw_line(
                    &self.input_buffer.buffer,
                    self.input_buffer.cursor_position
                );

            } else {
                
                drop(keyboard_buffer);
                x86_64::instructions::hlt();
            }
        }
        
        Ok(ExitCode::Success)
    }

    fn execute_command(&mut self, command: &str, args: &[String]) -> ShellResult {
        serial_println!("Shell: Executing command '{}' with args {:?}", command, args);
    
        let allocator_check = {
            let allocator = ALLOCATOR.lock();
            allocator.check_allocation_state()
        };
        
        if !allocator_check {
            serial_println!("Warning: Memory allocator in potentially invalid state");
            return Err(ShellError::InternalError);
        }

        let current_pid = {
            let process_list = PROCESS_LIST.lock();
            process_list.current().map(|p| p.id())
        };
        
        let result = match self.executor.execute(command, args) {
            Ok(exit_code) => {
                match exit_code {
                    ExitCode::Success => Ok(ExitCode::Success),
                    ExitCode::CommandNotFound => Ok(ExitCode::CommandNotFound),
                    _ => {
                        serial_println!("Command exited with status: {:?}", exit_code);
                        Ok(exit_code)
                    }
                }
            },
            Err(e) => {
                self.display.display_error(&e);
                Err(e)
            }
        };

        if let Some(pid) = current_pid {
            let mut process_list = PROCESS_LIST.lock();
            if process_list.current().map(|p| p.id()) != Some(pid) {
                if let Some(current) = process_list.get_mut_by_id(pid) {
                    current.set_state(ProcessState::Running);
                }
            }
        }
    
        result
    }

    fn handle_input(&mut self, byte: u8) -> ShellResult {
        match byte {
            
            b'\n' => {
                self.input_buffer.buffer.push(b'\n');
                WRITER.lock().write_byte(b'\n');
                let command = String::from_utf8_lossy(&self.input_buffer.buffer).into_owned();
                self.input_buffer.buffer.clear();
                self.input_buffer.cursor_position = 0;
                self.execute_command(&command, &[])
            },
    
            
            8 | 127 => {
                if self.input_buffer.cursor_position > 0 {
                    let mut writer = WRITER.lock();
                    if writer.column_position > 0 {
                        writer.column_position -= 1;
                        writer.write_byte(b' ');
                        writer.column_position -= 1;
                    }
                    
                    self.input_buffer.buffer.remove(self.input_buffer.cursor_position - 1);
                    self.input_buffer.cursor_position -= 1;
                }
                Ok(ExitCode::Success)
            },
    
            
            27 => {
                let mut keyboard_buffer = KEYBOARD_BUFFER.lock();
                if keyboard_buffer.len() >= 2 {
                    let seq1 = keyboard_buffer[0];
                    let seq2 = keyboard_buffer[1];
                    keyboard_buffer.drain(0..2);
                    drop(keyboard_buffer);
                    
                    if seq1 == b'[' {
                        match seq2 {
                            b'D' => { 
                                if self.input_buffer.move_cursor_left() {
                                    self.display.redraw_line(
                                        &self.input_buffer.buffer,
                                        self.input_buffer.cursor_position
                                    );
                                }
                                Ok(ExitCode::Success)
                            },
                            b'C' => { 
                                if self.input_buffer.move_cursor_right() {
                                    self.display.redraw_line(
                                        &self.input_buffer.buffer,
                                        self.input_buffer.cursor_position
                                    );
                                }
                                Ok(ExitCode::Success)
                            },
                            b'A' => { 
                                if self.history_position > 0 {
                                    self.history_position -= 1;
                                    if let Some(previous_cmd) = self.history.get(self.history_position) {
                                        self.input_buffer.clear();
                                        self.input_buffer.buffer.extend(previous_cmd.bytes());
                                        self.input_buffer.cursor_position = previous_cmd.len();
                                        self.display.redraw_line(
                                            &self.input_buffer.buffer,
                                            self.input_buffer.cursor_position
                                        );
                                    }
                                }
                                Ok(ExitCode::Success)
                            },
                            b'B' => { 
                                if self.history_position < self.history.len() {
                                    self.history_position += 1;
                                    let cmd = if self.history_position == self.history.len() {
                                        ""
                                    } else {
                                        &self.history[self.history_position]
                                    };
                                    self.input_buffer.clear();
                                    self.input_buffer.buffer.extend(cmd.bytes());
                                    self.input_buffer.cursor_position = cmd.len();
                                    self.display.redraw_line(
                                        &self.input_buffer.buffer,
                                        self.input_buffer.cursor_position
                                    );
                                }
                                Ok(ExitCode::Success)
                            },
                            _ => Ok(ExitCode::Success)
                        }
                    } else {
                        Ok(ExitCode::Success)
                    }
                } else {
                    drop(keyboard_buffer);
                    Ok(ExitCode::Success)
                }
            },
    
            
            32..=126 => {
                if self.input_buffer.buffer.len() < 1024 {
                    self.input_buffer.buffer.insert(self.input_buffer.cursor_position, byte);
                    self.input_buffer.cursor_position += 1;
                    WRITER.lock().write_byte(byte);
                }
                Ok(ExitCode::Success)
            },
    
            _ => Ok(ExitCode::Success), 
        }
    }
    
    fn display_error(&self, error: &ShellError) {
        let message = match error {
            ShellError::CommandNotFound => self.display.display_error(&ShellError::CommandNotFound),
            ShellError::InvalidArguments => self.display.display_error(&ShellError::InvalidArguments),
            ShellError::ExecutionFailed => self.display.display_error(&ShellError::ExecutionFailed),
            ShellError::IOError => self.display.display_error(&ShellError::IOError),
            ShellError::PermissionDenied => self.display.display_error(&ShellError::PermissionDenied),
            ShellError::PathNotFound => self.display.display_error(&ShellError::PathNotFound),
            ShellError::InvalidPath => self.display.display_error(&ShellError::InvalidPath),
            ShellError::EnvironmentError => self.display.display_error(&ShellError::EnvironmentError),
            ShellError::InternalError => self.display.display_error(&ShellError::InternalError),
            ShellError::BufferOverflow => self.display.display_error(&ShellError::BufferOverflow),
            ShellError::SyntaxError => self.display.display_error(&ShellError::SyntaxError),
            ShellError::NotADirectory => self.display.display_error(&ShellError::NotADirectory),
        };
    }

    fn update_current_dir(&mut self, new_path: String) -> ShellResult {
        let mut fs = FILESYSTEM.lock();
        match validate_path(&mut *fs, &new_path) {
            Ok(_) => {
                self.current_dir = normalize_path(&new_path);
                Ok(ExitCode::Success)
            },
            Err(FsError::NotADirectory) => Err(ShellError::NotADirectory),
            Err(FsError::NotFound) => Err(ShellError::PathNotFound),
            Err(_) => Err(ShellError::InvalidPath),
        }
    }

    fn refresh_prompt(&mut self) {
        let current_dir = {
            let process_list = PROCESS_LIST.lock();
            process_list.current()
                .map(|p| p.current_dir.clone())
                .unwrap_or_else(|| String::from("/"))
        };
        self.display.set_prompt(format!("{}> ", current_dir));
    }
    
    fn redraw_line(&mut self) {
        let mut writer = WRITER.lock();
        let prompt_len = self.current_dir.len() + 2; 
        
        
        writer.column_position = 0;
        for _ in 0..BUFFER_WIDTH {
            writer.write_byte(b' ');
        }
        
        
        writer.column_position = 0;
        write!(writer, "{}> ", self.current_dir).unwrap();
        
        
        for (i, &byte) in self.input_buffer.buffer.iter().enumerate() {
            if i == self.input_buffer.cursor_position {
                
                writer.color_code = ColorCode::new(Color::Black, Color::White);
            }
            writer.write_byte(byte);
            writer.color_code = ColorCode::new(Color::White, Color::Black);
        }
        
        
        writer.column_position = prompt_len + self.input_buffer.cursor_position;
    }

    pub fn shutdown(&mut self) -> ShellResult {
        self.is_running = false;
        Ok(ExitCode::Success)
    }
}