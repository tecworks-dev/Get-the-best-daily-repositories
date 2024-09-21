/*
This file is part of GigaPaste.

GigaPaste is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

GigaPaste is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with GigaPaste. If not, see <https://www.gnu.org/licenses/>.
*/
function copyToClipboard(text){

	if (navigator.clipboard && navigator.clipboard.writeText) {

		navigator.clipboard.writeText(text)

	}else{

		//sometimes clipboard api has problems on mobile browsers or won't work unless https connection, fallback
		//into this method instead
		let element = document.createElement("textarea");
		element.value = text;
		document.body.appendChild(element);
		element.select();
		document.execCommand("copy");
		document.body.removeChild(element);

	}

}
