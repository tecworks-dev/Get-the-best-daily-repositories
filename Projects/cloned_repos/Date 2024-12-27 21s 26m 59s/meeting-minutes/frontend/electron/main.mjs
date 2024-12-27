import { app, BrowserWindow } from 'electron';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let mainWindow;
let loadURL;

function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function initializeApp() {
    const serve = await import('electron-serve');
    loadURL = serve.default({ directory: 'out' });
}

async function createWindow() {
    if (!loadURL) {
        await initializeApp();
    }

    // Create the browser window.
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        titleBarStyle: 'hiddenInset',
        trafficLightPosition: { x: 20, y: 20 },
        backgroundColor: '#ffffff',
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js')
        }
    });

    // In development, load from Next.js dev server
    if (process.env.NODE_ENV === 'development') {
        // Wait for Next.js server to be ready
        await wait(2000);
        
        try {
            await mainWindow.loadURL('http://localhost:3000');
            mainWindow.webContents.openDevTools();
        } catch (error) {
            console.error('Failed to load URL:', error);
            // Retry loading the URL
            await wait(2000);
            try {
                await mainWindow.loadURL('http://localhost:3000');
            } catch (retryError) {
                console.error('Failed to load URL after retry:', retryError);
            }
        }

        // Handle page load failures
        mainWindow.webContents.on('did-fail-load', async () => {
            console.log('Page failed to load, retrying...');
            await wait(1000);
            mainWindow.loadURL('http://localhost:3000');
        });
    } else {
        // In production, load the built app using electron-serve
        await loadURL(mainWindow);
    }
}

// This method will be called when Electron has finished initialization
app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});