"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.log = void 0;
const tslib_1 = require("tslib");
const express_1 = tslib_1.__importDefault(require("express"));
const immich_1 = tslib_1.__importDefault(require("./immich"));
const render_1 = tslib_1.__importDefault(require("./render"));
const dayjs_1 = tslib_1.__importDefault(require("dayjs"));
const types_1 = require("./types");
require('dotenv').config();
const app = (0, express_1.default)();
// Add the EJS view engine, to render the gallery page
app.set('view engine', 'ejs');
// Serve static assets from the /public folder
app.use(express_1.default.static('public'));
// An incoming request for a shared link
app.get('/share/:key', (req, res) => tslib_1.__awaiter(void 0, void 0, void 0, function* () {
    res.set('Cache-Control', 'public, max-age=' + process.env.CACHE_AGE);
    if (!immich_1.default.isKey(req.params.key)) {
        (0, exports.log)('Invalid share key ' + req.params.key);
        res.status(404).send();
    }
    else {
        const sharedLink = yield immich_1.default.getShareByKey(req.params.key);
        if (!sharedLink) {
            (0, exports.log)('Unknown share key ' + req.params.key);
            res.status(404).send();
        }
        else if (!sharedLink.assets.length) {
            (0, exports.log)('No assets for key ' + req.params.key);
            res.status(404).send();
        }
        else if (sharedLink.assets.length === 1) {
            // This is an individual item (not a gallery)
            (0, exports.log)('Serving link ' + req.params.key);
            const asset = sharedLink.assets[0];
            if (asset.type === types_1.AssetType.image) {
                // For photos, output the image directly
                yield render_1.default.assetBuffer(res, sharedLink.assets[0], getSize(req));
            }
            else if (asset.type === types_1.AssetType.video) {
                // For videos, show the video as a web player
                yield render_1.default.gallery(res, sharedLink, 1);
            }
        }
        else {
            // Multiple images - render as a gallery
            (0, exports.log)('Serving link ' + req.params.key);
            yield render_1.default.gallery(res, sharedLink);
        }
    }
}));
// Output the buffer data for a photo or video
app.get('/:type(photo|video)/:key/:id', (req, res) => tslib_1.__awaiter(void 0, void 0, void 0, function* () {
    res.set('Cache-Control', 'public, max-age=' + process.env.CACHE_AGE);
    // Check for valid key and ID
    if (immich_1.default.isKey(req.params.key) && immich_1.default.isId(req.params.id)) {
        // Check if the key is a valid share link
        const sharedLink = yield immich_1.default.getShareByKey(req.params.key);
        if (sharedLink === null || sharedLink === void 0 ? void 0 : sharedLink.assets.length) {
            // Check that the requested asset exists in this share
            const asset = sharedLink.assets.find(x => x.id === req.params.id);
            if (asset) {
                asset.type = req.params.type === 'video' ? types_1.AssetType.video : types_1.AssetType.image;
                render_1.default.assetBuffer(res, asset, getSize(req)).then();
                return;
            }
        }
    }
    (0, exports.log)('No asset found for ' + req.path);
    res.status(404).send();
}));
// Send a 404 for all other routes
app.get('*', (req, res) => {
    (0, exports.log)('Invalid route ' + req.path);
    res.status(404).send();
});
/**
 * Sanitise the data for an incoming query string `size` parameter
 * e.g. https://example.com/share/abc...xyz?size=thumbnail
 */
const getSize = (req) => {
    var _a;
    return ((_a = req === null || req === void 0 ? void 0 : req.query) === null || _a === void 0 ? void 0 : _a.size) === 'thumbnail' ? types_1.ImageSize.thumbnail : types_1.ImageSize.original;
};
/**
 * Output a console.log message with timestamp
 */
const log = (message) => console.log((0, dayjs_1.default)().format() + ' ' + message);
exports.log = log;
app.listen(3000, () => {
    console.log((0, dayjs_1.default)().format() + ' Server started');
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaW5kZXguanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi9zcmMvaW5kZXgudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7OztBQUFBLDhEQUE2QjtBQUM3Qiw4REFBNkI7QUFDN0IsOERBQTZCO0FBQzdCLDBEQUF5QjtBQUN6QixtQ0FBOEM7QUFHOUMsT0FBTyxDQUFDLFFBQVEsQ0FBQyxDQUFDLE1BQU0sRUFBRSxDQUFBO0FBRTFCLE1BQU0sR0FBRyxHQUFHLElBQUEsaUJBQU8sR0FBRSxDQUFBO0FBQ3JCLHNEQUFzRDtBQUN0RCxHQUFHLENBQUMsR0FBRyxDQUFDLGFBQWEsRUFBRSxLQUFLLENBQUMsQ0FBQTtBQUM3Qiw4Q0FBOEM7QUFDOUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxpQkFBTyxDQUFDLE1BQU0sQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFBO0FBRWpDLHdDQUF3QztBQUN4QyxHQUFHLENBQUMsR0FBRyxDQUFDLGFBQWEsRUFBRSxDQUFPLEdBQUcsRUFBRSxHQUFHLEVBQUUsRUFBRTtJQUN4QyxHQUFHLENBQUMsR0FBRyxDQUFDLGVBQWUsRUFBRSxrQkFBa0IsR0FBRyxPQUFPLENBQUMsR0FBRyxDQUFDLFNBQVMsQ0FBQyxDQUFBO0lBQ3BFLElBQUksQ0FBQyxnQkFBTSxDQUFDLEtBQUssQ0FBQyxHQUFHLENBQUMsTUFBTSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUM7UUFDbEMsSUFBQSxXQUFHLEVBQUMsb0JBQW9CLEdBQUcsR0FBRyxDQUFDLE1BQU0sQ0FBQyxHQUFHLENBQUMsQ0FBQTtRQUMxQyxHQUFHLENBQUMsTUFBTSxDQUFDLEdBQUcsQ0FBQyxDQUFDLElBQUksRUFBRSxDQUFBO0lBQ3hCLENBQUM7U0FBTSxDQUFDO1FBQ04sTUFBTSxVQUFVLEdBQUcsTUFBTSxnQkFBTSxDQUFDLGFBQWEsQ0FBQyxHQUFHLENBQUMsTUFBTSxDQUFDLEdBQUcsQ0FBQyxDQUFBO1FBQzdELElBQUksQ0FBQyxVQUFVLEVBQUUsQ0FBQztZQUNo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LEdBQUcsRUFBRTtJQUNwQixPQUFPLENBQUMsR0FBRyxDQUFDLElBQUEsZUFBSyxHQUFFLENBQUMsTUFBTSxFQUFFLEdBQUcsaUJBQWlCLENBQUMsQ0FBQTtBQUNuRCxDQUFDLENBQUMsQ0FBQSIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBleHByZXNzIGZyb20gJ2V4cHJlc3MnXG5pbXBvcnQgaW1taWNoIGZyb20gJy4vaW1taWNoJ1xuaW1wb3J0IHJlbmRlciBmcm9tICcuL3JlbmRlcidcbmltcG9ydCBkYXlqcyBmcm9tICdkYXlqcydcbmltcG9ydCB7IEFzc2V0VHlwZSwgSW1hZ2VTaXplIH0gZnJvbSAnLi90eXBlcydcbmltcG9ydCB7IFJlcXVlc3QgfSBmcm9tICdleHByZXNzLXNlcnZlLXN0YXRpYy1jb3JlJ1xuXG5yZXF1aXJlKCdkb3RlbnYnKS5jb25maWcoKVxuXG5jb25zdCBhcHAgPSBleHByZXNzKClcbi8vIEFkZCB0aGUgRUpTIHZpZXcgZW5naW5lLCB0byByZW5kZXIgdGhlIGdhbGxlcnkgcGFnZVxuYXBwLnNldCgndmlldyBlbmdpbmUnLCAnZWpzJylcbi8vIFNlcnZlIHN0YXRpYyBhc3NldHMgZnJvbSB0aGUgL3B1YmxpYyBmb2xkZXJcbmFwcC51c2UoZXhwcmVzcy5zdGF0aWMoJ3B1YmxpYycpKVxuXG4vLyBBbiBpbmNvbWluZyByZXF1ZXN0IGZvciBhIHNoYXJlZCBsaW5rXG5hcHAuZ2V0KCcvc2hhcmUvOmtleScsIGFzeW5jIChyZXEsIHJlcykgPT4ge1xuICByZXMuc2V0KCdDYWNoZS1Db250cm9sJywgJ3B1YmxpYywgbWF4LWFnZT0nICsgcHJvY2Vzcy5lbnYuQ0FDSEVfQUdFKVxuICBpZiAoIWltbWljaC5pc0tleShyZXEucGFyYW1zLmtleSkpIHtcbiAgICBsb2coJ0ludmFsaWQgc2hhcmUga2V5ICcgKyByZXEucGFyYW1zLmtleSlcbiAgICByZXMuc3RhdHVzKDQwNCkuc2VuZCgpXG4gIH0gZWxzZSB7XG4gICAgY29uc3Qgc2hhcmVkTGluayA9IGF3YWl0IGltbWljaC5nZXRTaGFyZUJ5S2V5KHJlcS5wYXJhbXMua2V5KVxuICAgIGlmICghc2hhcmVkTGluaykge1xuICAgICAgbG9nKCdVbmtub3duIHNoYXJlIGtleSAnICsgcmVxLnBhcmFtcy5rZXkpXG4gICAgICByZXMuc3RhdHVzKDQwNCkuc2VuZCgpXG4gICAgfSBlbHNlIGlmICghc2hhcmVkTGluay5hc3NldHMubGVuZ3RoKSB7XG4gICAgICBsb2coJ05vIGFzc2V0cyBmb3Iga2V5ICcgKyByZXEucGFyYW1zLmtleSlcbiAgICAgIHJlcy5zdGF0dXMoNDA0KS5zZW5kKClcbiAgICB9IGVsc2UgaWYgKHNoYXJlZExpbmsuYXNzZXRzLmxlbmd0aCA9PT0gMSkge1xuICAgICAgLy8gVGhpcyBpcyBhbiBpbmRpdmlkdWFsIGl0ZW0gKG5vdCBhIGdhbGxlcnkpXG4gICAgICBsb2coJ1NlcnZpbmcgbGluayAnICsgcmVxLnBhcmFtcy5rZXkpXG4gICAgICBjb25zdCBhc3NldCA9IHNoYXJlZExpbmsuYXNzZXRzWzBdXG4gICAgICBpZiAoYXNzZXQudHlwZSA9PT0gQXNzZXRUeXBlLmltYWdlKSB7XG4gICAgICAgIC8vIEZvciBwaG90b3MsIG91dHB1dCB0aGUgaW1hZ2UgZGlyZWN0bHlcbiAgICAgICAgYXdhaXQgcmVuZGVyLmFzc2V0QnVmZmVyKHJlcywgc2hhcmVkTGluay5hc3NldHNbMF0sIGdldFNpemUocmVxKSlcbiAgICAgIH0gZWxzZSBpZiAoYXNzZXQudHlwZSA9PT0gQXNzZXRUeXBlLnZpZGVvKSB7XG4gICAgICAgIC8vIEZvciB2aWRlb3MsIHNob3cgdGhlIHZpZGVvIGFzIGEgd2ViIHBsYXllclxuICAgICAgICBhd2FpdCByZW5kZXIuZ2FsbGVyeShyZXMsIHNoYXJlZExpbmssIDEpXG4gICAgICB9XG4gICAgfSBlbHNlIHtcbiAgICAgIC8vIE11bHRpcGxlIGltYWdlcyAtIHJlbmRlciBhcyBhIGdhbGxlcnlcbiAgICAgIGxvZygnU2VydmluZyBsaW5rICcgKyByZXEucGFyYW1zLmtleSlcbiAgICAgIGF3YWl0IHJlbmRlci5nYWxsZXJ5KHJlcywgc2hhcmVkTGluaylcbiAgICB9XG4gIH1cbn0pXG5cbi8vIE91dHB1dCB0aGUgYnVmZmVyIGRhdGEgZm9yIGEgcGhvdG8gb3IgdmlkZW9cbmFwcC5nZXQoJy86dHlwZShwaG90b3x2aWRlbykvOmtleS86aWQnLCBhc3luYyAocmVxLCByZXMpID0+IHtcbiAgcmVzLnNldCgnQ2FjaGUtQ29udHJvbCcsICdwdWJsaWMsIG1heC1hZ2U9JyArIHByb2Nlc3MuZW52LkNBQ0hFX0FHRSlcbiAgLy8gQ2hlY2sgZm9yIHZhbGlkIGtleSBhbmQgSURcbiAgaWYgKGltbWljaC5pc0tleShyZXEucGFyYW1zLmtleSkgJiYgaW1taWNoLmlzSWQocmVxLnBhcmFtcy5pZCkpIHtcbiAgICAvLyBDaGVjayBpZiB0aGUga2V5IGlzIGEgdmFsaWQgc2hhcmUgbGlua1xuICAgIGNvbnN0IHNoYXJlZExpbmsgPSBhd2FpdCBpbW1pY2guZ2V0U2hhcmVCeUtleShyZXEucGFyYW1zLmtleSlcbiAgICBpZiAoc2hhcmVkTGluaz8uYXNzZXRzLmxlbmd0aCkge1xuICAgICAgLy8gQ2hlY2sgdGhhdCB0aGUgcmVxdWVzdGVkIGFzc2V0IGV4aXN0cyBpbiB0aGlzIHNoYXJlXG4gICAgICBjb25zdCBhc3NldCA9IHNoYXJlZExpbmsuYXNzZXRzLmZpbmQoeCA9PiB4LmlkID09PSByZXEucGFyYW1zLmlkKVxuICAgICAgaWYgKGFzc2V0KSB7XG4gICAgICAgIGFzc2V0LnR5cGUgPSByZXEucGFyYW1zLnR5cGUgPT09ICd2aWRlbycgPyBBc3NldFR5cGUudmlkZW8gOiBBc3NldFR5cGUuaW1hZ2VcbiAgICAgICAgcmVuZGVyLmFzc2V0QnVmZmVyKHJlcywgYXNzZXQsIGdldFNpemUocmVxKSkudGhlbigpXG4gICAgICAgIHJldHVyblxuICAgICAgfVxuICAgIH1cbiAgfVxuICBsb2coJ05vIGFzc2V0IGZvdW5kIGZvciAnICsgcmVxLnBhdGgpXG4gIHJlcy5zdGF0dXMoNDA0KS5zZW5kKClcbn0pXG5cbi8vIFNlbmQgYSA0MDQgZm9yIGFsbCBvdGhlciByb3V0ZXNcbmFwcC5nZXQoJyonLCAocmVxLCByZXMpID0+IHtcbiAgbG9nKCdJbnZhbGlkIHJvdXRlICcgKyByZXEucGF0aClcbiAgcmVzLnN0YXR1cyg0MDQpLnNlbmQoKVxufSlcblxuLyoqXG4gKiBTYW5pdGlzZSB0aGUgZGF0YSBmb3IgYW4gaW5jb21pbmcgcXVlcnkgc3RyaW5nIGBzaXplYCBwYXJhbWV0ZXJcbiAqIGUuZy4gaHR0cHM6Ly9leGFtcGxlLmNvbS9zaGFyZS9hYmMuLi54eXo/c2l6ZT10aHVtYm5haWxcbiAqL1xuY29uc3QgZ2V0U2l6ZSA9IChyZXE6IFJlcXVlc3QpID0+IHtcbiAgcmV0dXJuIHJlcT8ucXVlcnk/LnNpemUgPT09ICd0aHVtYm5haWwnID8gSW1hZ2VTaXplLnRodW1ibmFpbCA6IEltYWdlU2l6ZS5vcmlnaW5hbFxufVxuXG4vKipcbiAqIE91dHB1dCBhIGNvbnNvbGUubG9nIG1lc3NhZ2Ugd2l0aCB0aW1lc3RhbXBcbiAqL1xuZXhwb3J0IGNvbnN0IGxvZyA9IChtZXNzYWdlOiBzdHJpbmcpID0+IGNvbnNvbGUubG9nKGRheWpzKCkuZm9ybWF0KCkgKyAnICcgKyBtZXNzYWdlKVxuXG5hcHAubGlzdGVuKDMwMDAsICgpID0+IHtcbiAgY29uc29sZS5sb2coZGF5anMoKS5mb3JtYXQoKSArICcgU2VydmVyIHN0YXJ0ZWQnKVxufSlcbiJdfQ==