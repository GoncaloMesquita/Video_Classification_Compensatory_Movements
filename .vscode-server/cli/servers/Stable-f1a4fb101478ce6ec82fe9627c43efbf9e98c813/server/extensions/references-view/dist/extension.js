(()=>{"use strict";var e={475:function(e,t,i){var n,r=this&&this.__createBinding||(Object.create?function(e,t,i,n){void 0===n&&(n=i);var r=Object.getOwnPropertyDescriptor(t,i);r&&!("get"in r?!t.__esModule:r.writable||r.configurable)||(r={enumerable:!0,get:function(){return t[i]}}),Object.defineProperty(e,n,r)}:function(e,t,i,n){void 0===n&&(n=i),e[n]=t[i]}),o=this&&this.__setModuleDefault||(Object.create?function(e,t){Object.defineProperty(e,"default",{enumerable:!0,value:t})}:function(e,t){e.default=t}),s=this&&this.__importStar||(n=function(e){return n=Object.getOwnPropertyNames||function(e){var t=[];for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&(t[t.length]=i);return t},n(e)},function(e){if(e&&e.__esModule)return e;var t={};if(null!=e)for(var i=n(e),s=0;s<i.length;s++)"default"!==i[s]&&r(t,e,i[s]);return o(t,e),t});Object.defineProperty(t,"__esModule",{value:!0}),t.register=function(e,t){const i=new h(t.workspaceState,0);function n(t,n){let r;i.value=t;const o=e.getInput();n instanceof l.CallItem?r=new l.CallsTreeInput(new a.Location(n.item.uri,n.item.selectionRange.start),i.value):o instanceof l.CallsTreeInput&&(r=new l.CallsTreeInput(o.location,i.value)),r&&e.setInput(r)}t.subscriptions.push(a.commands.registerCommand("references-view.showCallHierarchy",(function(){if(a.window.activeTextEditor){const t=new l.CallsTreeInput(new a.Location(a.window.activeTextEditor.document.uri,a.window.activeTextEditor.selection.active),i.value);e.setInput(t)}})),a.commands.registerCommand("references-view.showOutgoingCalls",(e=>n(1,e))),a.commands.registerCommand("references-view.showIncomingCalls",(e=>n(0,e))),a.commands.registerCommand("references-view.removeCallItem",u))};const a=s(i(398)),c=i(376),l=i(170);function u(e){e instanceof l.CallItem&&e.remove()}class h{constructor(e,t=1){this._mem=e,this._value=t,this._ctxMode=new c.ContextKey("references-view.callHierarchyMode");const i=e.get(h._key);this.value="number"==typeof i&&i>=0&&i<=1?i:t}get value(){return this._value}set value(e){this._value=e,this._ctxMode.set(0===this._value?"showIncoming":"showOutgoing"),this._mem.update(h._key,e)}}h._key="references-view.callHierarchyMode"},170:function(e,t,i){var n,r=this&&this.__createBinding||(Object.create?function(e,t,i,n){void 0===n&&(n=i);var r=Object.getOwnPropertyDescriptor(t,i);r&&!("get"in r?!t.__esModule:r.writable||r.configurable)||(r={enumerable:!0,get:function(){return t[i]}}),Object.defineProperty(e,n,r)}:function(e,t,i,n){void 0===n&&(n=i),e[n]=t[i]}),o=this&&this.__setModuleDefault||(Object.create?function(e,t){Object.defineProperty(e,"default",{enumerable:!0,value:t})}:function(e,t){e.default=t}),s=this&&this.__importStar||(n=function(e){return n=Object.getOwnPropertyNames||function(e){var t=[];for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&(t[t.length]=i);return t},n(e)},function(e){if(e&&e.__esModule)return e;var t={};if(null!=e)for(var i=n(e),s=0;s<i.length;s++)"default"!==i[s]&&r(t,e,i[s]);return o(t,e),t});Object.defineProperty(t,"__esModule",{value:!0}),t.CallItem=t.CallsTreeInput=void 0;const a=s(i(398)),c=i(376);class l{constructor(e,t){this.location=e,this.direction=t,this.contextValue="callHierarchy",this.title=0===t?a.l10n.t("Callers Of"):a.l10n.t("Calls From")}async resolve(){const e=await Promise.resolve(a.commands.executeCommand("vscode.prepareCallHierarchy",this.location.uri,this.location.range.start)),t=new h(this.direction,e??[]),i=new d(t);if(0!==t.roots.length)return{provider:i,get message(){return 0===t.roots.length?a.l10n.t("No results."):void 0},navigation:t,highlights:t,dnd:t,dispose(){i.dispose()}}}with(e){return new l(e,this.direction)}}t.CallsTreeInput=l;class u{constructor(e,t,i,n){this.model=e,this.item=t,this.parent=i,this.locations=n}remove(){this.model.remove(this)}}t.CallItem=u;class h{constructor(e,t){this.direction=e,this.roots=[],this._onDidChange=new a.EventEmitter,this.onDidChange=this._onDidChange.event,this.roots=t.map((e=>new u(this,e,void 0,void 0)))}async _resolveCalls(e){if(0===this.direction){const t=await a.commands.executeCommand("vscode.provideIncomingCalls",e.item);return t?t.map((t=>new u(this,t.from,e,t.fromRanges.map((e=>new a.Location(t.from.uri,e)))))):[]}{const t=await a.commands.executeCommand("vscode.provideOutgoingCalls",e.item);return t?t.map((t=>new u(this,t.to,e,t.fromRanges.map((t=>new a.Location(e.item.uri,t)))))):[]}}async getCallChildren(e){return e.children||(e.children=await this._resolveCalls(e)),e.children}location(e){return new a.Location(e.item.uri,e.item.range)}nearest(e,t){return this.roots.find((t=>t.item.uri.toString()===e.toString()))??this.roots[0]}next(e){return this._move(e,!0)??e}previous(e){return this._move(e,!1)??e}_move(e,t){if(e.children?.length)return t?e.children[0]:(0,c.tail)(e.children);const i=this.roots.includes(e)?this.roots:e.parent?.children;if(i?.length){const n=i.indexOf(e);return i[n+(t?1:-1)+i.length%i.length]}}getDragUri(e){return(0,c.asResourceUrl)(e.item.uri,e.item.range)}getEditorHighlights(e,t){return e.locations?e.locations.filter((e=>e.uri.toString()===t.toString())).map((e=>e.range)):e.item.uri.toString()===t.toString()?[e.item.selectionRange]:void 0}remove(e){const t=this.roots.includes(e)?this.roots:e.parent?.children;t&&((0,c.del)(t,e),this._onDidChange.fire(this))}}class d{constructor(e){this._model=e,this._emitter=new a.EventEmitter,this.onDidChangeTreeData=this._emitter.event,this._modelListener=e.onDidChange((e=>this._emitter.fire(e instanceof u?e:void 0)))}dispose(){this._emitter.dispose(),this._modelListener.dispose()}getTreeItem(e){const t=new a.TreeItem(e.item.name);let i;if(t.description=e.item.detail,t.tooltip=t.label&&e.item.detail?`${t.label} - ${e.item.detail}`:t.label?`${t.label}`:e.item.detail,t.contextValue="call-item",t.iconPath=(0,c.getThemeIcon)(e.item.kind),1===e.model.direction)i=[e.item.uri,{selection:e.item.selectionRange.with({end:e.item.selectionRange.start})}];else{let t;if(e.locations)for(const i of e.locations)i.uri.toString()===e.item.uri.toString()&&(t=t?.isBefore(i.range.start)?t:i.range.start);t||(t=e.item.selectionRange.start),i=[e.item.uri,{selection:new a.Range(t,t)}]}return t.command={command:"vscode.open",title:a.l10n.t("Open Call"),arguments:i},t.collapsibleState=a.TreeItemCollapsibleState.Collapsed,t}getChildren(e){return e?this._model.getCallChildren(e):this._model.roots}getParent(e){return e.parent}}},256:function(e,t,i){var n,r=this&&this.__createBinding||(Object.create?function(e,t,i,n){void 0===n&&(n=i);var r=Object.getOwnPropertyDescriptor(t,i);r&&!("get"in r?!t.__esModule:r.writable||r.configurable)||(r={enumerable:!0,get:function(){return t[i]}}),Object.defineProperty(e,n,r)}:function(e,t,i,n){void 0===n&&(n=i),e[n]=t[i]}),o=this&&this.__setModuleDefault||(Object.create?function(e,t){Object.defineProperty(e,"default",{enumerable:!0,value:t})}:function(e,t){e.default=t}),s=this&&this.__importStar||(n=function(e){return n=Object.getOwnPropertyNames||function(e){var t=[];for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&(t[t.length]=i);return t},n(e)},function(e){if(e&&e.__esModule)return e;var t={};if(null!=e)for(var i=n(e),s=0;s<i.length;s++)"default"!==i[s]&&r(t,e,i[s]);return o(t,e),t});Object.defineProperty(t,"__esModule",{value:!0}),t.activate=function(e){const t=new l.SymbolsTree;return c.register(t,e),a.register(t,e),u.register(t,e),{setInput:function(e){t.setInput(e)},getInput:function(){return t.getInput()}}};const a=s(i(475)),c=s(i(464)),l=i(541),u=s(i(227))},76:function(e,t,i){var n,r=this&&this.__createBinding||(Object.create?function(e,t,i,n){void 0===n&&(n=i);var r=Object.getOwnPropertyDescriptor(t,i);r&&!("get"in r?!t.__esModule:r.writable||r.configurable)||(r={enumerable:!0,get:function(){return t[i]}}),Object.defineProperty(e,n,r)}:function(e,t,i,n){void 0===n&&(n=i),e[n]=t[i]}),o=this&&this.__setModuleDefault||(Object.create?function(e,t){Object.defineProperty(e,"default",{enumerable:!0,value:t})}:function(e,t){e.default=t}),s=this&&this.__importStar||(n=function(e){return n=Object.getOwnPropertyNames||function(e){var t=[];for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&(t[t.length]=i);return t},n(e)},function(e){if(e&&e.__esModule)return e;var t={};if(null!=e)for(var i=n(e),s=0;s<i.length;s++)"default"!==i[s]&&r(t,e,i[s]);return o(t,e),t});Object.defineProperty(t,"__esModule",{value:!0}),t.EditorHighlights=void 0;const a=s(i(398));t.EditorHighlights=class{constructor(e,t){this._view=e,this._delegate=t,this._decorationType=a.window.createTextEditorDecorationType({backgroundColor:new a.ThemeColor("editor.findMatchHighlightBackground"),rangeBehavior:a.DecorationRangeBehavior.ClosedClosed,overviewRulerLane:a.OverviewRulerLane.Center,overviewRulerColor:new a.ThemeColor("editor.findMatchHighlightBackground")}),this.disposables=[],this._ignore=new Set,this.disposables.push(a.workspace.onDidChangeTextDocument((e=>this._ignore.add(e.document.uri.toString()))),a.window.onDidChangeActiveTextEditor((()=>e.visible&&this.update())),e.onDidChangeVisibility((e=>e.visible?this._show():this._hide())),e.onDidChangeSelection((()=>{e.visible&&this.update()}))),this._show()}dispose(){a.Disposable.from(...this.disposables).dispose();for(const e of a.window.visibleTextEditors)e.setDecorations(this._decorationType,[])}_show(){const{activeTextEditor:e}=a.window;if(!e||!e.viewColumn)return;if(this._ignore.has(e.document.uri.toString()))return;const[t]=this._view.selection;if(!t)return;const i=this._delegate.getEditorHighlights(t,e.document.uri);i&&e.setDecorations(this._decorationType,i)}_hide(){for(const e of a.window.visibleTextEditors)e.setDecorations(this._decorationType,[])}update(){this._hide(),this._show()}}},87:function(e,t,i){var n,r=this&&this.__createBinding||(Object.create?function(e,t,i,n){void 0===n&&(n=i);var r=Object.getOwnPropertyDescriptor(t,i);r&&!("get"in r?!t.__esModule:r.writable||r.configurable)||(r={enumerable:!0,get:function(){return t[i]}}),Object.defineProperty(e,n,r)}:function(e,t,i,n){void 0===n&&(n=i),e[n]=t[i]}),o=this&&this.__setModuleDefault||(Object.create?function(e,t){Object.defineProperty(e,"default",{enumerable:!0,value:t})}:function(e,t){e.default=t}),s=this&&this.__importStar||(n=function(e){return n=Object.getOwnPropertyNames||function(e){var t=[];for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&(t[t.length]=i);return t},n(e)},function(e){if(e&&e.__esModule)return e;var t={};if(null!=e)for(var i=n(e),s=0;s<i.length;s++)"default"!==i[s]&&r(t,e,i[s]);return o(t,e),t});Object.defineProperty(t,"__esModule",{value:!0}),t.Navigation=void 0;const a=s(i(398)),c=i(376);t.Navigation=class{constructor(e){this._view=e,this._disposables=[],this._ctxCanNavigate=new c.ContextKey("references-view.canNavigate"),this._disposables.push(a.commands.registerCommand("references-view.next",(()=>this.next(!1))),a.commands.registerCommand("references-view.prev",(()=>this.previous(!1))))}dispose(){a.Disposable.from(...this._disposables).dispose()}update(e){this._delegate=e,this._ctxCanNavigate.set(Boolean(this._delegate))}_anchor(){if(!this._delegate)return;const[e]=this._view.selection;return e||(a.window.activeTextEditor?this._delegate.nearest(a.window.activeTextEditor.document.uri,a.window.activeTextEditor.selection.active):void 0)}_open(e,t){a.commands.executeCommand("vscode.open",e.uri,{selection:new a.Selection(e.range.start,e.range.start),preserveFocus:t})}previous(e){if(!this._delegate)return;const t=this._anchor();if(!t)return;const i=this._delegate.previous(t),n=this._delegate.location(i);n&&(this._view.reveal(i,{select:!0,focus:!0}),this._open(n,e))}next(e){if(!this._delegate)return;const t=this._anchor();if(!t)return;const i=this._delegate.next(t),n=this._delegate.location(i);n&&(this._view.reveal(i,{select:!0,focus:!0}),this._open(n,e))}}},464:function(e,t,i){var n,r=this&&this.__createBinding||(Object.create?function(e,t,i,n){void 0===n&&(n=i);var r=Object.getOwnPropertyDescriptor(t,i);r&&!("get"in r?!t.__esModule:r.writable||r.configurable)||(r={enumerable:!0,get:function(){return t[i]}}),Object.defineProperty(e,n,r)}:function(e,t,i,n){void 0===n&&(n=i),e[n]=t[i]}),o=this&&this.__setModuleDefault||(Object.create?function(e,t){Object.defineProperty(e,"default",{enumerable:!0,value:t})}:function(e,t){e.default=t}),s=this&&this.__importStar||(n=function(e){return n=Object.getOwnPropertyNames||function(e){var t=[];for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&(t[t.length]=i);return t},n(e)},function(e){if(e&&e.__esModule)return e;var t={};if(null!=e)for(var i=n(e),s=0;s<i.length;s++)"default"!==i[s]&&r(t,e,i[s]);return o(t,e),t});Object.defineProperty(t,"__esModule",{value:!0}),t.register=function(e,t){function i(t,i){if(a.window.activeTextEditor){const n=new c.ReferencesTreeInput(t,new a.Location(a.window.activeTextEditor.document.uri,a.window.activeTextEditor.selection.active),i);e.setInput(n)}}let n;t.subscriptions.push(a.commands.registerCommand("references-view.findReferences",(()=>i("References","vscode.executeReferenceProvider"))),a.commands.registerCommand("references-view.findImplementations",(()=>i("Implementations","vscode.executeImplementationProvider"))),a.commands.registerCommand("references-view.find",((...e)=>a.commands.executeCommand("references-view.findReferences",...e))),a.commands.registerCommand("references-view.removeReferenceItem",u),a.commands.registerCommand("references-view.copy",h),a.commands.registerCommand("references-view.copyAll",l),a.commands.registerCommand("references-view.copyPath",d));const r="references.preferredLocation";function o(t){if(t&&!t.affectsConfiguration(r))return;const i=a.workspace.getConfiguration().get(r);n?.dispose(),n=void 0,"view"===i&&(n=a.commands.registerCommand("editor.action.showReferences",(async(t,i,n)=>{const r=new c.ReferencesTreeInput(a.l10n.t("References"),new a.Location(t,i),"vscode.executeReferenceProvider",n);e.setInput(r)})))}t.subscriptions.push(a.workspace.onDidChangeConfiguration(o)),t.subscriptions.push({dispose:()=>n?.dispose()}),o()};const a=s(i(398)),c=i(769),l=async e=>{e instanceof c.ReferenceItem?h(e.file.model):e instanceof c.FileItem&&h(e.model)};function u(e){(e instanceof c.FileItem||e instanceof c.ReferenceItem)&&e.remove()}async function h(e){let t;(e instanceof c.ReferencesModel||e instanceof c.ReferenceItem||e instanceof c.FileItem)&&(t=await e.asCopyText()),t&&await a.env.clipboard.writeText(t)}async function d(e){e instanceof c.FileItem&&("file"===e.uri.scheme?a.env.clipboard.writeText(e.uri.fsPath):a.env.clipboard.writeText(e.uri.toString(!0)))}},769:function(e,t,i){var n,r=this&&this.__createBinding||(Object.create?function(e,t,i,n){void 0===n&&(n=i);var r=Object.getOwnPropertyDescriptor(t,i);r&&!("get"in r?!t.__esModule:r.writable||r.configurable)||(r={enumerable:!0,get:function(){return t[i]}}),Object.defineProperty(e,n,r)}:function(e,t,i,n){void 0===n&&(n=i),e[n]=t[i]}),o=this&&this.__setModuleDefault||(Object.create?function(e,t){Object.defineProperty(e,"default",{enumerable:!0,value:t})}:function(e,t){e.default=t}),s=this&&this.__importStar||(n=function(e){return n=Object.getOwnPropertyNames||function(e){var t=[];for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&(t[t.length]=i);return t},n(e)},function(e){if(e&&e.__esModule)return e;var t={};if(null!=e)for(var i=n(e),s=0;s<i.length;s++)"default"!==i[s]&&r(t,e,i[s]);return o(t,e),t});Object.defineProperty(t,"__esModule",{value:!0}),t.ReferenceItem=t.FileItem=t.ReferencesModel=t.ReferencesTreeInput=void 0;const a=s(i(398)),c=i(376);class l{constructor(e,t,i,n){this.title=e,this.location=t,this._command=i,this._result=n,this.contextValue=i}async resolve(){let e;if(this._result)e=new u(this._result);else{const t=await Promise.resolve(a.commands.executeCommand(this._command,this.location.uri,this.location.range.start));e=new u(t??[])}if(0===e.items.length)return;const t=new h(e);return{provider:t,get message(){return e.message},navigation:e,highlights:e,dnd:e,dispose(){t.dispose()}}}with(e){return new l(this.title,e,this._command)}}t.ReferencesTreeInput=l;class u{constructor(e){let t;this._onDidChange=new a.EventEmitter,this.onDidChangeTreeData=this._onDidChange.event,this.items=[];for(const i of e.sort(u._compareLocations)){const e=i instanceof a.Location?i:new a.Location(i.targetUri,i.targetRange);t&&0===u._compareUriIgnoreFragment(t.uri,e.uri)||(t=new d(e.uri.with({fragment:""}),[],this),this.items.push(t)),t.references.push(new f(e,t))}}static _compareUriIgnoreFragment(e,t){const i=e.with({fragment:""}).toString(),n=t.with({fragment:""}).toString();return i<n?-1:i>n?1:0}static _compareLocations(e,t){const i=e instanceof a.Location?e.uri:e.targetUri,n=t instanceof a.Location?t.uri:t.targetUri;if(i.toString()<n.toString())return-1;if(i.toString()>n.toString())return 1;const r=e instanceof a.Location?e.range:e.targetRange,o=t instanceof a.Location?t.range:t.targetRange;return r.start.isBefore(o.start)?-1:r.start.isAfter(o.start)?1:0}get message(){if(0===this.items.length)return a.l10n.t("No results.");const e=this.items.reduce(((e,t)=>e+t.references.length),0),t=this.items.length;return 1===e&&1===t?a.l10n.t("{0} result in {1} file",e,t):1===e?a.l10n.t("{0} result in {1} files",e,t):1===t?a.l10n.t("{0} results in {1} file",e,t):a.l10n.t("{0} results in {1} files",e,t)}location(e){return e instanceof f?e.location:new a.Location(e.uri,e.references[0]?.location.range??new a.Position(0,0))}nearest(e,t){if(0===this.items.length)return;for(const i of this.items)if(i.uri.toString()===e.toString()){for(const e of i.references)if(e.location.range.contains(t))return e;let e;for(const n of i.references){if(n.location.range.end.isAfter(t))return n;e=n}if(e)return e;break}let i=0;const n=u._prefixLen(this.items[i].toString(),e.toString());for(let t=1;t<this.items.length;t++)u._prefixLen(this.items[t].uri.toString(),e.toString())>n&&(i=t);return this.items[i].references[0]}static _prefixLen(e,t){let i=0;for(;i<e.length&&i<t.length&&e.charCodeAt(i)===t.charCodeAt(i);)i+=1;return i}next(e){return this._move(e,!0)??e}previous(e){return this._move(e,!1)??e}_move(e,t){const i=t?1:-1,n=e=>{const t=(this.items.indexOf(e)+i+this.items.length)%this.items.length;return this.items[t]};if(e instanceof d)return t?n(e).references[0]:(0,c.tail)(n(e).references);if(e instanceof f){const t=e.file.references.indexOf(e)+i;return t<0?(0,c.tail)(n(e.file).references):t>=e.file.references.length?n(e.file).references[0]:e.file.references[t]}}getEditorHighlights(e,t){const i=this.items.find((e=>e.uri.toString()===t.toString()));return i?.references.map((e=>e.location.range))}remove(e){e instanceof d?((0,c.del)(this.items,e),this._onDidChange.fire(void 0)):((0,c.del)(e.file.references,e),0===e.file.references.length?((0,c.del)(this.items,e.file),this._onDidChange.fire(void 0)):this._onDidChange.fire(e.file))}async asCopyText(){let e="";for(const t of this.items)e+=`${await t.asCopyText()}\n`;return e}getDragUri(e){return e instanceof d?e.uri:(0,c.asResourceUrl)(e.file.uri,e.location.range)}}t.ReferencesModel=u;class h{constructor(e){this._model=e,this._onDidChange=new a.EventEmitter,this.onDidChangeTreeData=this._onDidChange.event,this._listener=e.onDidChangeTreeData((()=>this._onDidChange.fire(void 0)))}dispose(){this._onDidChange.dispose(),this._listener.dispose()}async getTreeItem(e){if(e instanceof d){const t=new a.TreeItem(e.uri);return t.contextValue="file-item",t.description=!0,t.iconPath=a.ThemeIcon.File,t.collapsibleState=a.TreeItemCollapsibleState.Collapsed,t}{const{range:t}=e.location,i=await e.getDocument(!0),{before:n,inside:r,after:o}=(0,c.getPreviewChunks)(i,t),s={label:n+r+o,highlights:[[n.length,n.length+r.length]]},l=new a.TreeItem(s);return l.collapsibleState=a.TreeItemCollapsibleState.None,l.contextValue="reference-item",l.command={command:"vscode.open",title:a.l10n.t("Open Reference"),arguments:[e.location.uri,{selection:t.with({end:t.start})}]},l}}async getChildren(e){return e?e instanceof d?e.references:void 0:this._model.items}getParent(e){return e instanceof f?e.file:void 0}}class d{constructor(e,t,i){this.uri=e,this.references=t,this.model=i}remove(){this.model.remove(this)}async asCopyText(){let e=`${a.workspace.asRelativePath(this.uri)}\n`;for(const t of this.references)e+=`  ${await t.asCopyText()}\n`;return e}}t.FileItem=d;class f{constructor(e,t){this.location=e,this.file=t}async getDocument(e){if(this._document||(this._document=a.workspace.openTextDocument(this.location.uri)),e){const e=this.file.model.next(this.file);e instanceof d&&e!==this.file?a.workspace.openTextDocument(e.uri):e instanceof f&&a.workspace.openTextDocument(e.location.uri)}return this._document}remove(){this.file.model.remove(this)}async asCopyText(){const e=await this.getDocument(),t=(0,c.getPreviewChunks)(e,this.location.range,21,!1);return`${this.location.range.start.line+1}, ${this.location.range.start.character+1}: ${t.before+t.inside+t.after}`}}t.ReferenceItem=f},541:function(e,t,i){var n,r=this&&this.__createBinding||(Object.create?function(e,t,i,n){void 0===n&&(n=i);var r=Object.getOwnPropertyDescriptor(t,i);r&&!("get"in r?!t.__esModule:r.writable||r.configurable)||(r={enumerable:!0,get:function(){return t[i]}}),Object.defineProperty(e,n,r)}:function(e,t,i,n){void 0===n&&(n=i),e[n]=t[i]}),o=this&&this.__setModuleDefault||(Object.create?function(e,t){Object.defineProperty(e,"default",{enumerable:!0,value:t})}:function(e,t){e.default=t}),s=this&&this.__importStar||(n=function(e){return n=Object.getOwnPropertyNames||function(e){var t=[];for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&(t[t.length]=i);return t},n(e)},function(e){if(e&&e.__esModule)return e;var t={};if(null!=e)for(var i=n(e),s=0;s<i.length;s++)"default"!==i[s]&&r(t,e,i[s]);return o(t,e),t});Object.defineProperty(t,"__esModule",{value:!0}),t.SymbolsTree=void 0;const a=s(i(398)),c=i(76),l=i(87),u=i(376);t.SymbolsTree=class{constructor(){this.viewId="references-view.tree",this._ctxIsActive=new u.ContextKey("reference-list.isActive"),this._ctxHasResult=new u.ContextKey("reference-list.hasResult"),this._ctxInputSource=new u.ContextKey("reference-list.source"),this._history=new m(this),this._provider=new h,this._dnd=new d,this._tree=a.window.createTreeView(this.viewId,{treeDataProvider:this._provider,showCollapseAll:!0,dragAndDropController:this._dnd}),this._navigation=new l.Navigation(this._tree)}dispose(){this._history.dispose(),this._tree.dispose(),this._sessionDisposable?.dispose()}getInput(){return this._input}async setInput(e){if(!await(0,u.isValidRequestPosition)(e.location.uri,e.location.range.start))return void this.clearInput();this._ctxInputSource.set(e.contextValue),this._ctxIsActive.set(!0),this._ctxHasResult.set(!0),a.commands.executeCommand(`${this.viewId}.focus`);const t=!this._input||Object.getPrototypeOf(this._input)!==Object.getPrototypeOf(e);this._input=e,this._sessionDisposable?.dispose(),this._tree.title=e.title,this._tree.message=t?void 0:this._tree.message;const i=Promise.resolve(e.resolve());this._provider.update(i.then((e=>e?.provider??this._history))),this._dnd.update(i.then((e=>e?.dnd)));const n=await i;if(this._input!==e)return;if(!n)return void this.clearInput();this._history.add(e),this._tree.message=n.message,this._navigation.update(n.navigation);const r=n.navigation?.nearest(e.location.uri,e.location.range.start);r&&this._tree.visible&&await this._tree.reveal(r,{select:!0,focus:!0,expand:!0});const o=[];let s;n.highlights&&(s=new c.EditorHighlights(this._tree,n.highlights),o.push(s)),n.provider.onDidChangeTreeData&&o.push(n.provider.onDidChangeTreeData((()=>{this._tree.title=e.title,this._tree.message=n.message,s?.update()}))),"function"==typeof n.dispose&&o.push(new a.Disposable((()=>n.dispose()))),this._sessionDisposable=a.Disposable.from(...o)}clearInput(){this._sessionDisposable?.dispose(),this._input=void 0,this._ctxHasResult.set(!1),this._ctxInputSource.reset(),this._tree.title=a.l10n.t("References"),this._tree.message=0===this._history.size?a.l10n.t("No results."):a.l10n.t("No results. Try running a previous search again:"),this._provider.update(Promise.resolve(this._history))}};class h{constructor(){this._onDidChange=new a.EventEmitter,this.onDidChangeTreeData=this._onDidChange.event}update(e){this._sessionDispoables?.dispose(),this._sessionDispoables=void 0,this._onDidChange.fire(void 0),this.provider=e,e.then((t=>{this.provider===e&&t.onDidChangeTreeData&&(this._sessionDispoables=t.onDidChangeTreeData(this._onDidChange.fire,this._onDidChange))})).catch((e=>{this.provider=void 0,console.error(e)}))}async getTreeItem(e){return this._assertProvider(),(await this.provider).getTreeItem(e)}async getChildren(e){return this._assertProvider(),(await this.provider).getChildren(e)}async getParent(e){this._assertProvider();const t=await this.provider;return t.getParent?t.getParent(e):void 0}_assertProvider(){if(!this.provider)throw new Error("MISSING provider")}}class d{constructor(){this.dropMimeTypes=[],this.dragMimeTypes=["text/uri-list"]}update(e){this._delegate=void 0,e.then((e=>this._delegate=e))}handleDrag(e,t){if(this._delegate){const i=[];for(const t of e){const e=this._delegate.getDragUri(t);e&&i.push(e.toString())}i.length>0&&t.set("text/uri-list",new a.DataTransferItem(i.join("\r\n")))}}handleDrop(){throw new Error("Method not implemented.")}}class f{constructor(e,t,i,n){this.key=e,this.word=t,this.anchor=i,this.input=n,this.description=`${a.workspace.asRelativePath(n.location.uri)} • ${n.title.toLocaleLowerCase()}`}}class m{constructor(e){this._tree=e,this._onDidChangeTreeData=new a.EventEmitter,this.onDidChangeTreeData=this._onDidChangeTreeData.event,this._disposables=[],this._ctxHasHistory=new u.ContextKey("reference-list.hasHistory"),this._inputs=new Map,this._disposables.push(a.commands.registerCommand("references-view.clear",(()=>e.clearInput())),a.commands.registerCommand("references-view.clearHistory",(()=>{this.clear(),e.clearInput()})),a.commands.registerCommand("references-view.refind",(e=>{e instanceof f&&this._reRunHistoryItem(e)})),a.commands.registerCommand("references-view.refresh",(()=>{const e=Array.from(this._inputs.values()).pop();e&&this._reRunHistoryItem(e)})),a.commands.registerCommand("_references-view.showHistoryItem",(async e=>{if(e instanceof f){const t=e.anchor.guessedTrackedPosition()??e.input.location.range.start;await a.commands.executeCommand("vscode.open",e.input.location.uri,{selection:new a.Range(t,t)})}})),a.commands.registerCommand("references-view.pickFromHistory",(async()=>{const e=(await this.getChildren()).map((e=>({label:e.word,description:e.description,item:e}))),t=await a.window.showQuickPick(e,{placeHolder:a.l10n.t("Select previous reference search")});t&&this._reRunHistoryItem(t.item)})))}dispose(){a.Disposable.from(...this._disposables).dispose(),this._onDidChangeTreeData.dispose()}_reRunHistoryItem(e){this._inputs.delete(e.key);const t=e.anchor.guessedTrackedPosition();let i=e.input;t&&!e.input.location.range.start.isEqual(t)&&(i=e.input.with(new a.Location(e.input.location.uri,t))),this._tree.setInput(i)}async add(e){const t=await a.workspace.openTextDocument(e.location.uri),i=new u.WordAnchor(t,e.location.range.start),n=t.getWordRangeAtPosition(e.location.range.start)??t.getWordRangeAtPosition(e.location.range.start,/[^\s]+/),r=n?t.getText(n):"???",o=new f(JSON.stringify([n?.start??e.location.range.start,e.location.uri,e.title]),r,i,e);this._inputs.delete(o.key),this._inputs.set(o.key,o),this._ctxHasHistory.set(!0)}clear(){this._inputs.clear(),this._ctxHasHistory.set(!1),this._onDidChangeTreeData.fire(void 0)}get size(){return this._inputs.size}getTreeItem(e){const t=new a.TreeItem(e.word);return t.description=e.description,t.command={command:"_references-view.showHistoryItem",arguments:[e],title:a.l10n.t("Rerun")},t.collapsibleState=a.TreeItemCollapsibleState.None,t.contextValue="history-item",t}getChildren(){return Promise.all([...this._inputs.values()].reverse())}getParent(){}}},227:function(e,t,i){var n,r=this&&this.__createBinding||(Object.create?function(e,t,i,n){void 0===n&&(n=i);var r=Object.getOwnPropertyDescriptor(t,i);r&&!("get"in r?!t.__esModule:r.writable||r.configurable)||(r={enumerable:!0,get:function(){return t[i]}}),Object.defineProperty(e,n,r)}:function(e,t,i,n){void 0===n&&(n=i),e[n]=t[i]}),o=this&&this.__setModuleDefault||(Object.create?function(e,t){Object.defineProperty(e,"default",{enumerable:!0,value:t})}:function(e,t){e.default=t}),s=this&&this.__importStar||(n=function(e){return n=Object.getOwnPropertyNames||function(e){var t=[];for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&(t[t.length]=i);return t},n(e)},function(e){if(e&&e.__esModule)return e;var t={};if(null!=e)for(var i=n(e),s=0;s<i.length;s++)"default"!==i[s]&&r(t,e,i[s]);return o(t,e),t});Object.defineProperty(t,"__esModule",{value:!0}),t.register=function(e,t){const i=new h(t.workspaceState,"subtypes");function n(t,n){let r;i.value=t;const o=e.getInput();n instanceof l.TypeItem?r=new l.TypesTreeInput(new a.Location(n.item.uri,n.item.selectionRange.start),i.value):n instanceof a.Location?r=new l.TypesTreeInput(n,i.value):o instanceof l.TypesTreeInput&&(r=new l.TypesTreeInput(o.location,i.value)),r&&e.setInput(r)}t.subscriptions.push(a.commands.registerCommand("references-view.showTypeHierarchy",(function(){if(a.window.activeTextEditor){const t=new l.TypesTreeInput(new a.Location(a.window.activeTextEditor.document.uri,a.window.activeTextEditor.selection.active),i.value);e.setInput(t)}})),a.commands.registerCommand("references-view.showSupertypes",(e=>n("supertypes",e))),a.commands.registerCommand("references-view.showSubtypes",(e=>n("subtypes",e))),a.commands.registerCommand("references-view.removeTypeItem",u))};const a=s(i(398)),c=i(376),l=i(10);function u(e){e instanceof l.TypeItem&&e.remove()}class h{constructor(e,t="subtypes"){this._mem=e,this._value=t,this._ctxMode=new c.ContextKey("references-view.typeHierarchyMode");const i=e.get(h._key);this.value="string"==typeof i?i:t}get value(){return this._value}set value(e){this._value=e,this._ctxMode.set(e),this._mem.update(h._key,e)}}h._key="references-view.typeHierarchyMode"},10:function(e,t,i){var n,r=this&&this.__createBinding||(Object.create?function(e,t,i,n){void 0===n&&(n=i);var r=Object.getOwnPropertyDescriptor(t,i);r&&!("get"in r?!t.__esModule:r.writable||r.configurable)||(r={enumerable:!0,get:function(){return t[i]}}),Object.defineProperty(e,n,r)}:function(e,t,i,n){void 0===n&&(n=i),e[n]=t[i]}),o=this&&this.__setModuleDefault||(Object.create?function(e,t){Object.defineProperty(e,"default",{enumerable:!0,value:t})}:function(e,t){e.default=t}),s=this&&this.__importStar||(n=function(e){return n=Object.getOwnPropertyNames||function(e){var t=[];for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&(t[t.length]=i);return t},n(e)},function(e){if(e&&e.__esModule)return e;var t={};if(null!=e)for(var i=n(e),s=0;s<i.length;s++)"default"!==i[s]&&r(t,e,i[s]);return o(t,e),t});Object.defineProperty(t,"__esModule",{value:!0}),t.TypeItem=t.TypesTreeInput=void 0;const a=s(i(398)),c=i(376);class l{constructor(e,t){this.location=e,this.direction=t,this.contextValue="typeHierarchy",this.title="supertypes"===t?a.l10n.t("Supertypes Of"):a.l10n.t("Subtypes Of")}async resolve(){const e=await Promise.resolve(a.commands.executeCommand("vscode.prepareTypeHierarchy",this.location.uri,this.location.range.start)),t=new h(this.direction,e??[]),i=new d(t);if(0!==t.roots.length)return{provider:i,get message(){return 0===t.roots.length?a.l10n.t("No results."):void 0},navigation:t,highlights:t,dnd:t,dispose(){i.dispose()}}}with(e){return new l(e,this.direction)}}t.TypesTreeInput=l;class u{constructor(e,t,i){this.model=e,this.item=t,this.parent=i}remove(){this.model.remove(this)}}t.TypeItem=u;class h{constructor(e,t){this.direction=e,this.roots=[],this._onDidChange=new a.EventEmitter,this.onDidChange=this._onDidChange.event,this.roots=t.map((e=>new u(this,e,void 0)))}async _resolveTypes(e){if("supertypes"===this.direction){const t=await a.commands.executeCommand("vscode.provideSupertypes",e.item);return t?t.map((t=>new u(this,t,e))):[]}{const t=await a.commands.executeCommand("vscode.provideSubtypes",e.item);return t?t.map((t=>new u(this,t,e))):[]}}async getTypeChildren(e){return e.children||(e.children=await this._resolveTypes(e)),e.children}getDragUri(e){return(0,c.asResourceUrl)(e.item.uri,e.item.range)}location(e){return new a.Location(e.item.uri,e.item.range)}nearest(e,t){return this.roots.find((t=>t.item.uri.toString()===e.toString()))??this.roots[0]}next(e){return this._move(e,!0)??e}previous(e){return this._move(e,!1)??e}_move(e,t){if(e.children?.length)return t?e.children[0]:(0,c.tail)(e.children);const i=this.roots.includes(e)?this.roots:e.parent?.children;if(i?.length){const n=i.indexOf(e);return i[n+(t?1:-1)+i.length%i.length]}}getEditorHighlights(e,t){return e.item.uri.toString()===t.toString()?[e.item.selectionRange]:void 0}remove(e){const t=this.roots.includes(e)?this.roots:e.parent?.children;t&&((0,c.del)(t,e),this._onDidChange.fire(this))}}class d{constructor(e){this._model=e,this._emitter=new a.EventEmitter,this.onDidChangeTreeData=this._emitter.event,this._modelListener=e.onDidChange((e=>this._emitter.fire(e instanceof u?e:void 0)))}dispose(){this._emitter.dispose(),this._modelListener.dispose()}getTreeItem(e){const t=new a.TreeItem(e.item.name);return t.description=e.item.detail,t.contextValue="type-item",t.iconPath=(0,c.getThemeIcon)(e.item.kind),t.command={command:"vscode.open",title:a.l10n.t("Open Type"),arguments:[e.item.uri,{selection:e.item.selectionRange.with({end:e.item.selectionRange.start})}]},t.collapsibleState=a.TreeItemCollapsibleState.Collapsed,t}getChildren(e){return e?this._model.getTypeChildren(e):this._model.roots}getParent(e){return e.parent}}},376:function(e,t,i){var n,r=this&&this.__createBinding||(Object.create?function(e,t,i,n){void 0===n&&(n=i);var r=Object.getOwnPropertyDescriptor(t,i);r&&!("get"in r?!t.__esModule:r.writable||r.configurable)||(r={enumerable:!0,get:function(){return t[i]}}),Object.defineProperty(e,n,r)}:function(e,t,i,n){void 0===n&&(n=i),e[n]=t[i]}),o=this&&this.__setModuleDefault||(Object.create?function(e,t){Object.defineProperty(e,"default",{enumerable:!0,value:t})}:function(e,t){e.default=t}),s=this&&this.__importStar||(n=function(e){return n=Object.getOwnPropertyNames||function(e){var t=[];for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&(t[t.length]=i);return t},n(e)},function(e){if(e&&e.__esModule)return e;var t={};if(null!=e)for(var i=n(e),s=0;s<i.length;s++)"default"!==i[s]&&r(t,e,i[s]);return o(t,e),t});Object.defineProperty(t,"__esModule",{value:!0}),t.WordAnchor=t.ContextKey=void 0,t.del=function(e,t){const i=e.indexOf(t);i>=0&&e.splice(i,1)},t.tail=function(e){return e[e.length-1]},t.asResourceUrl=function(e,t){return e.with({fragment:`L${1+t.start.line},${1+t.start.character}-${1+t.end.line},${1+t.end.character}`})},t.isValidRequestPosition=async function(e,t){const i=await a.workspace.openTextDocument(e);let n=i.getWordRangeAtPosition(t);return n||(n=i.getWordRangeAtPosition(t,/[^\s]+/)),Boolean(n)},t.getPreviewChunks=function(e,t,i=8,n=!0){const r=t.start.with({character:Math.max(0,t.start.character-i)}),o=e.getWordRangeAtPosition(r);let s=e.getText(new a.Range(o?o.start:r,t.start));const c=e.getText(t),l=t.end.translate(0,331);let u=e.getText(new a.Range(t.end,l));return n&&(s=s.replace(/^\s*/g,""),u=u.replace(/\s*$/g,"")),{before:s,inside:c,after:u}},t.getThemeIcon=function(e){const t=c[e];return t?new a.ThemeIcon(t):void 0};const a=s(i(398));t.ContextKey=class{constructor(e){this.name=e}async set(e){await a.commands.executeCommand("setContext",this.name,e)}async reset(){await a.commands.executeCommand("setContext",this.name,void 0)}},t.WordAnchor=class{constructor(e,t){this._doc=e,this._position=t,this._version=e.version,this._word=this._getAnchorWord(e,t)}_getAnchorWord(e,t){const i=e.getWordRangeAtPosition(t)||e.getWordRangeAtPosition(t,/[^\s]+/);return i&&e.getText(i)}guessedTrackedPosition(){if(!this._word)return this._position;if(this._version===this._doc.version)return this._position;const e=this._getAnchorWord(this._doc,this._position);if(this._word===e)return this._position;const t=this._position.line;let i,n,r=0;do{if(n=!1,i=t+r,i<this._doc.lineCount){n=!0;const e=this._doc.lineAt(i).text.indexOf(this._word);if(e>=0)return new a.Position(i,e)}if(r+=1,i=t-r,i>=0){n=!0;const e=this._doc.lineAt(i).text.indexOf(this._word);if(e>=0)return new a.Position(i,e)}}while(r<100&&n);return this._position}};const c=["symbol-file","symbol-module","symbol-namespace","symbol-package","symbol-class","symbol-method","symbol-property","symbol-field","symbol-constructor","symbol-enum","symbol-interface","symbol-function","symbol-variable","symbol-constant","symbol-string","symbol-number","symbol-boolean","symbol-array","symbol-object","symbol-key","symbol-null","symbol-enum-member","symbol-struct","symbol-event","symbol-operator","symbol-type-parameter"]},398:e=>{e.exports=require("vscode")}},t={},i=function i(n){var r=t[n];if(void 0!==r)return r.exports;var o=t[n]={exports:{}};return e[n].call(o.exports,o,o.exports,i),o.exports}(256),n=exports;for(var r in i)n[r]=i[r];i.__esModule&&Object.defineProperty(n,"__esModule",{value:!0})})();
//# sourceMappingURL=https://main.vscode-cdn.net/sourcemaps/f1a4fb101478ce6ec82fe9627c43efbf9e98c813/extensions/references-view/dist/extension.js.map