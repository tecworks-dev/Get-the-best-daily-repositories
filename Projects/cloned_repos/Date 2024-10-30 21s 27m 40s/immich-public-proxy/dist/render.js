"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const tslib_1 = require("tslib");
const immich_1 = tslib_1.__importDefault(require("./immich"));
const types_1 = require("./types");
class Render {
    assetBuffer(res, asset, size) {
        return tslib_1.__awaiter(this, void 0, void 0, function* () {
            const data = yield immich_1.default.getAssetBuffer(asset, size);
            if (data) {
                for (const header of ['content-type', 'content-length']) {
                    res.set(header, data.headers[header]);
                }
                res.send(Buffer.from(yield data.arrayBuffer()));
            }
            else {
                res.status(404).send();
            }
        });
    }
    /**
     * Render a gallery page for a given SharedLink, using EJS and lightGallery.
     *
     * @param res - ExpressJS Response
     * @param share - Immich `shared-link` containing the assets to show in the gallery
     * @param [openItem] - Immediately open a lightbox to the Nth item when the gallery loads
     */
    gallery(res, share, openItem) {
        return tslib_1.__awaiter(this, void 0, void 0, function* () {
            const items = [];
            for (const asset of share.assets) {
                let video;
                if (asset.type === types_1.AssetType.video) {
                    // Populate the data-video property
                    video = JSON.stringify({
                        source: [
                            {
                                src: immich_1.default.videoUrl(share.key, asset.id),
                                type: yield immich_1.default.getContentType(asset)
                            }
                        ],
                        attributes: {
                            preload: false,
                            controls: true
                        }
                    });
                }
                items.push({
                    originalUrl: immich_1.default.photoUrl(share.key, asset.id),
                    thumbnailUrl: immich_1.default.photoUrl(share.key, asset.id, types_1.ImageSize.thumbnail),
                    video
                });
            }
            res.render('gallery', {
                items,
                openItem
            });
        });
    }
}
const render = new Render();
exports.default = render;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmVuZGVyLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vc3JjL3JlbmRlci50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7QUFBQSw4REFBNkI7QUFFN0IsbUNBQWlFO0FBRWpFLE1BQU0sTUFBTTtJQUNKLFdBQVcsQ0FBRSxHQUFhLEVBQUUsS0FBWSxFQUFFLElBQWdCOztZQUM5RCxNQUFNLElBQUksR0FBRyxNQUFNLGdCQUFNLENBQUMsY0FBYyxDQUFDLEtBQUssRUFBRSxJQUFJLENBQUMsQ0FBQTtZQUNyRCxJQUFJLElBQUksRUFBRSxDQUFDO2dCQUNULEtBQUssTUFBTSxNQUFNLElBQUksQ0FBQyxjQUFjLEVBQUUsZ0JBQWdCLENBQUMsRUFBRSxDQUFDO29CQUN4RCxHQUFHLENBQUMsR0FBRyxDQUFDLE1BQU0sRUFBRSxJQUFJLENBQUMsT0FBTyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUE7Z0JBQ3ZDLENBQUM7Z0JBQ0QsR0FBRyxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsSUFBSSxDQUFDLE1BQU0sSUFBSSxDQUFDLFdBQVcsRUFBRSxDQUFDLENBQUMsQ0FBQTtZQUNqRCxDQUFDO2lCQUFNLENBQUM7Z0JBQ04sR0FBRyxDQUFDLE1BQU0sQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLEVBQUUsQ0FBQTtZQUN4QixDQUFDO1FBQ0gsQ0FBQztLQUFBO0lBRUQ7Ozs7OztPQU1HO0lBQ0csT0FBTyxDQUFFLEdBQWEsRUFBRSxLQUFpQixFQUFFLFFBQWlCOztZQUNoRSxNQUFNLEtBQUssR0FBRyxFQUFFLENBQUE7WUFDaEIsS0FBSyxNQUFNLEtBQUssSUFBSSxLQUFLLENBQUMsTUFBTSxFQUFFLENBQUM7Z0JBQ2pDLElBQUksS0FBSyxDQUFBO2dCQUNULElBQUksS0FBSyxDQUFDLElBQUksS0FBSyxpQkFBUyxDQUFDLEtBQUssRUFBRSxDQUFDO29CQUNuQyxtQ0FBbUM7b0JBQ25DLEtBQUssR0FBRyxJQUFJLENBQUMsU0FBUyxDQUFDO3dCQUNyQixNQUFNLEVBQUU7NEJBQ047Z0NBQ0UsR0FBRyxFQUFFLGdCQUFNLENBQUMsUUFBUSxDQUFDLEtBQUssQ0FBQyxHQUFHLEVBQUUsS0FBSyxDQUFDLEVBQUUsQ0FBQztnQ0FDekMsSUFBSSxFQUFFLE1BQU0sZ0JBQU0sQ0FBQyxjQUFjLENBQUMsS0FBSyxDQUFDOzZCQUN6Qzt5QkFDRjt3QkFDRCxVQUFVLEVBQUU7NEJBQ1YsT0FBTyxFQUFFLEtBQUs7NEJBQ2QsUUFBUSxFQUFFLElBQUk7eUJBQ2Y7cUJBQ0YsQ0FBQyxDQUFBO2dCQUNKLENBQUM7Z0JBQ0QsS0FBSyxDQUFDLElBQUksQ0FBQztvQkFDVCxXQUFXLEVBQUUsZ0JBQU0sQ0FBQyxRQUFRLENBQUMsS0FBSyxDQUFDLEdBQUcsRUFBRSxLQUFLLENBQUMsRUFBRSxDQUFDO29CQUNqRCxZQUFZLEVBQUUsZ0JBQU0sQ0FBQyxRQUFRLENBQUMsS0FBSyxDQUFDLEdBQUcsRUFBRSxLQUFLLENBQUMsRUFBRSxFQUFFLGlCQUFTLENBQUMsU0FBUyxDQUFDO29CQUN2RSxLQUFLO2lCQUNOLENBQUMsQ0FBQTtZQUNKLENBQUM7WUFDRCxHQUFHLENBQUMsTUFBTSxDQUFDLFNBQVMsRUFBRTtnQkFDcEIsS0FBSztnQkFDTCxRQUFRO2FBQ1QsQ0FBQyxDQUFBO1FBQ0osQ0FBQztLQUFBO0NBQ0Y7QUFFRCxNQUFNLE1BQU0sR0FBRyxJQUFJLE1BQU0sRUFBRSxDQUFBO0FBRTNCLGtCQUFlLE1BQU0sQ0FBQSIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBpbW1pY2ggZnJvbSAnLi9pbW1pY2gnXG5pbXBvcnQgeyBSZXNwb25zZSB9IGZyb20gJ2V4cHJlc3Mtc2VydmUtc3RhdGljLWNvcmUnXG5pbXBvcnQgeyBBc3NldCwgQXNzZXRUeXBlLCBJbWFnZVNpemUsIFNoYXJlZExpbmsgfSBmcm9tICcuL3R5cGVzJ1xuXG5jbGFzcyBSZW5kZXIge1xuICBhc3luYyBhc3NldEJ1ZmZlciAocmVzOiBSZXNwb25zZSwgYXNzZXQ6IEFzc2V0LCBzaXplPzogSW1hZ2VTaXplKSB7XG4gICAgY29uc3QgZGF0YSA9IGF3YWl0IGltbWljaC5nZXRBc3NldEJ1ZmZlcihhc3NldCwgc2l6ZSlcbiAgICBpZiAoZGF0YSkge1xuICAgICAgZm9yIChjb25zdCBoZWFkZXIgb2YgWydjb250ZW50LXR5cGUnLCAnY29udGVudC1sZW5ndGgnXSkge1xuICAgICAgICByZXMuc2V0KGhlYWRlciwgZGF0YS5oZWFkZXJzW2hlYWRlcl0pXG4gICAgICB9XG4gICAgICByZXMuc2VuZChCdWZmZXIuZnJvbShhd2FpdCBkYXRhLmFycmF5QnVmZmVyKCkpKVxuICAgIH0gZWxzZSB7XG4gICAgICByZXMuc3RhdHVzKDQwNCkuc2VuZCgpXG4gICAgfVxuICB9XG5cbiAgLyoqXG4gICAqIFJlbmRlciBhIGdhbGxlcnkgcGFnZSBmb3IgYSBnaXZlbiBTaGFyZWRMaW5rLCB1c2luZyBFSlMgYW5kIGxpZ2h0R2FsbGVyeS5cbiAgICpcbiAgICogQHBhcmFtIHJlcyAtIEV4cHJlc3NKUyBSZXNwb25zZVxuICAgKiBAcGFyYW0gc2hhcmUgLSBJbW1pY2ggYHNoYXJlZC1saW5rYCBjb250YWluaW5nIHRoZSBhc3NldHMgdG8gc2hvdyBpbiB0aGUgZ2FsbGVyeVxuICAgKiBAcGFyYW0gW29wZW5JdGVtXSAtIEltbWVkaWF0ZWx5IG9wZW4gYSBsaWdodGJveCB0byB0aGUgTnRoIGl0ZW0gd2hlbiB0aGUgZ2FsbGVyeSBsb2Fkc1xuICAgKi9cbiAgYXN5bmMgZ2FsbGVyeSAocmVzOiBSZXNwb25zZSwgc2hhcmU6IFNoYXJlZExpbmssIG9wZW5JdGVtPzogbnVtYmVyKSB7XG4gICAgY29uc3QgaXRlbXMgPSBbXVxuICAgIGZvciAoY29uc3QgYXNzZXQgb2Ygc2hhcmUuYXNzZXRzKSB7XG4gICAgICBsZXQgdmlkZW9cbiAgICAgIGlmIChhc3NldC50eXBlID09PSBBc3NldFR5cGUudmlkZW8pIHtcbiAgICAgICAgLy8gUG9wdWxhdGUgdGhlIGRhdGEtdmlkZW8gcHJvcGVydHlcbiAgICAgICAgdmlkZW8gPSBKU09OLnN0cmluZ2lmeSh7XG4gICAgICAgICAgc291cmNlOiBbXG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgIHNyYzogaW1taWNoLnZpZGVvVXJsKHNoYXJlLmtleSwgYXNzZXQuaWQpLFxuICAgICAgICAgICAgICB0eXBlOiBhd2FpdCBpbW1pY2guZ2V0Q29udGVudFR5cGUoYXNzZXQpXG4gICAgICAgICAgICB9XG4gICAgICAgICAgXSxcbiAgICAgICAgICBhdHRyaWJ1dGVzOiB7XG4gICAgICAgICAgICBwcmVsb2FkOiBmYWxzZSxcbiAgICAgICAgICAgIGNvbnRyb2xzOiB0cnVlXG4gICAgICAgICAgfVxuICAgICAgICB9KVxuICAgICAgfVxuICAgICAgaXRlbXMucHVzaCh7XG4gICAgICAgIG9yaWdpbmFsVXJsOiBpbW1pY2gucGhvdG9Vcmwoc2hhcmUua2V5LCBhc3NldC5pZCksXG4gICAgICAgIHRodW1ibmFpbFVybDogaW1taWNoLnBob3RvVXJsKHNoYXJlLmtleSwgYXNzZXQuaWQsIEltYWdlU2l6ZS50aHVtYm5haWwpLFxuICAgICAgICB2aWRlb1xuICAgICAgfSlcbiAgICB9XG4gICAgcmVzLnJlbmRlcignZ2FsbGVyeScsIHtcbiAgICAgIGl0ZW1zLFxuICAgICAgb3Blbkl0ZW1cbiAgICB9KVxuICB9XG59XG5cbmNvbnN0IHJlbmRlciA9IG5ldyBSZW5kZXIoKVxuXG5leHBvcnQgZGVmYXVsdCByZW5kZXJcbiJdfQ==