(self.webpackChunkjupyter_renderers=self.webpackChunkjupyter_renderers||[]).push([[270],{34546:(t,e,r)=>{var n=r(692),o=r(43769);Object.defineProperty(e,"__esModule",{value:!0}),Object.defineProperty(e,"objectToReactElement",{enumerable:!0,get:function(){return d.objectToReactElement}}),e.default=void 0;var i=o(r(11486)),a=o(r(57864)),u=o(r(38455)),s=o(r(12285)),f=o(r(6582)),c=o(r(5816)),l=n(r(96540)),p=r(2543),d=r(94723),v=function(t){function e(){return(0,i.default)(this,e),(0,u.default)(this,(0,s.default)(e).apply(this,arguments))}return(0,f.default)(e,t),(0,a.default)(e,[{key:"shouldComponentUpdate",value:function(t){return t.data!==this.props.data}},{key:"render",value:function(){try{var t=(0,p.cloneDeep)(this.props.data);return(0,d.objectToReactElement)(t)}catch(t){return l.createElement(l.Fragment,null,l.createElement("pre",{style:{backgroundColor:"ghostwhite",color:"black",fontWeight:"600",display:"block",padding:"10px",marginBottom:"20px"}},"There was an error rendering VDOM data from the kernel or notebook"),l.createElement("code",null,t.toString()))}}}]),e}(l.Component);e.default=v,(0,c.default)(v,"MIMETYPE","application/vdom.v1+json")},94723:(t,e,r)=>{var n=r(43769);Object.defineProperty(e,"__esModule",{value:!0}),e.objectToReactElement=s,e.arrayToReactChildren=f;var o=n(r(803)),i=n(r(68632)),a=n(r(73146)),u=r(96540);function s(t){var e=[];if(!t.tagName||"string"!=typeof t.tagName)throw new Error("Invalid tagName on ".concat((0,a.default)(t,null,2)));if(!t.attributes||(0,i.default)(t.attributes)||"object"!==(0,o.default)(t.attributes))throw new Error("Attributes must exist on a VDOM Object as an object");if(null===t.attributes.style||void 0===t.attributes.style);else if((0,i.default)(t.attributes.style)||"object"!==(0,o.default)(t.attributes.style))throw new Error("Style attribute must be an object like { 'backgroundColor': 'DeepPink' }");e[0]=t.tagName,e[1]=t.attributes;var r=t.children;if(r)if((0,i.default)(r))void 0===e[1]&&(e[1]=null),e=e.concat(f(r));else if("string"==typeof r)e[2]=r;else{if("object"!==(0,o.default)(r))throw new Error("children of a vdom element must be a string, object, null, or array of vdom nodes");e[2]=s(r)}return u.createElement.apply({},e)}function f(t){for(var e=[],r=0,n=t.length;r<n;r++){var a=t[r];if(null!==a)if((0,i.default)(a))e.push(f(a));else if("string"==typeof a)e.push(a);else{if("object"!==(0,o.default)(a))throw new Error('invalid vdom child: "'.concat(a,'"'));var u={tagName:a.tagName,attributes:a.attributes,children:a.children,key:r};a.attributes&&a.attributes.key&&(u.key=a.attributes.key),e.push(s(u))}}return e}},37149:(t,e,r)=>{r(18355),t.exports=r(46438).Array.isArray},85079:(t,e,r)=>{var n=r(46438),o=n.JSON||(n.JSON={stringify:JSON.stringify});t.exports=function(t){return o.stringify.apply(o,arguments)}},80040:(t,e,r)=>{r(78978),r(7914),r(77789),r(78353),r(50836),t.exports=r(46438).WeakMap},2832:t=>{t.exports=function(t,e,r,n){if(!(t instanceof e)||void 0!==n&&n in t)throw TypeError(r+": incorrect invocation!");return t}},48219:(t,e,r)=>{var n=r(58852),o=r(7001),i=r(66310),a=r(70181),u=r(99244);t.exports=function(t,e){var r=1==t,s=2==t,f=3==t,c=4==t,l=6==t,p=5==t||l,d=e||u;return function(e,u,v){for(var h,y,b=i(e),g=o(b),m=n(u,v,3),w=a(g.length),x=0,_=r?d(e,w):s?d(e,0):void 0;w>x;x++)if((p||x in g)&&(y=m(h=g[x],x,b),t))if(r)_[x]=y;else if(y)switch(t){case 3:return!0;case 5:return h;case 6:return x;case 2:_.push(h)}else if(c)return!1;return l?-1:f||c?c:_}}},49742:(t,e,r)=>{var n=r(24401),o=r(15461),i=r(30254)("species");t.exports=function(t){var e;return o(t)&&("function"!=typeof(e=t.constructor)||e!==Array&&!o(e.prototype)||(e=void 0),n(e)&&null===(e=e[i])&&(e=void 0)),void 0===e?Array:e}},99244:(t,e,r)=>{var n=r(49742);t.exports=function(t,e){return new(n(t))(e)}},2162:(t,e,r)=>{var n=r(30953),o=r(75172).getWeak,i=r(80812),a=r(24401),u=r(2832),s=r(95838),f=r(48219),c=r(75509),l=r(6096),p=f(5),d=f(6),v=0,h=function(t){return t._l||(t._l=new y)},y=function(){this.a=[]},b=function(t,e){return p(t.a,(function(t){return t[0]===e}))};y.prototype={get:function(t){var e=b(this,t);if(e)return e[1]},has:function(t){return!!b(this,t)},set:function(t,e){var r=b(this,t);r?r[1]=e:this.a.push([t,e])},delete:function(t){var e=d(this.a,(function(e){return e[0]===t}));return~e&&this.a.splice(e,1),!!~e}},t.exports={getConstructor:function(t,e,r,i){var f=t((function(t,n){u(t,f,e,"_i"),t._t=e,t._i=v++,t._l=void 0,null!=n&&s(n,r,t[i],t)}));return n(f.prototype,{delete:function(t){if(!a(t))return!1;var r=o(t);return!0===r?h(l(this,e)).delete(t):r&&c(r,this._i)&&delete r[this._i]},has:function(t){if(!a(t))return!1;var r=o(t);return!0===r?h(l(this,e)).has(t):r&&c(r,this._i)}}),f},def:function(t,e,r){var n=o(i(e),!0);return!0===n?h(t).set(e,r):n[t._i]=r,t},ufstore:h}},99373:(t,e,r)=>{var n=r(66670),o=r(88535),i=r(75172),a=r(81984),u=r(2677),s=r(30953),f=r(95838),c=r(2832),l=r(24401),p=r(1356),d=r(78423).f,v=r(48219)(0),h=r(58219);t.exports=function(t,e,r,y,b,g){var m=n[t],w=m,x=b?"set":"add",_=w&&w.prototype,E={};return h&&"function"==typeof w&&(g||_.forEach&&!a((function(){(new w).entries().next()})))?(w=e((function(e,r){c(e,w,t,"_c"),e._c=new m,null!=r&&f(r,b,e[x],e)})),v("add,clear,delete,forEach,get,has,set,keys,values,entries,toJSON".split(","),(function(t){var e="add"==t||"set"==t;!(t in _)||g&&"clear"==t||u(w.prototype,t,(function(r,n){if(c(this,w,t),!e&&g&&!l(r))return"get"==t&&void 0;var o=this._c[t](0===r?0:r,n);return e?this:o}))})),g||d(w.prototype,"size",{get:function(){return this._c.size}})):(w=y.getConstructor(e,t,b,x),s(w.prototype,r),i.NEED=!0),p(w,t),E[t]=w,o(o.G+o.W+o.F,E),g||y.setStrong(w,t,b),w}},95838:(t,e,r)=>{var n=r(58852),o=r(27904),i=r(12828),a=r(80812),u=r(70181),s=r(55298),f={},c={},l=t.exports=function(t,e,r,l,p){var d,v,h,y,b=p?function(){return t}:s(t),g=n(r,l,e?2:1),m=0;if("function"!=typeof b)throw TypeError(t+" is not iterable!");if(i(b)){for(d=u(t.length);d>m;m++)if((y=e?g(a(v=t[m])[0],v[1]):g(t[m]))===f||y===c)return y}else for(h=b.call(t);!(v=h.next()).done;)if((y=o(h,g,v.value,e))===f||y===c)return y};l.BREAK=f,l.RETURN=c},12828:(t,e,r)=>{var n=r(20210),o=r(30254)("iterator"),i=Array.prototype;t.exports=function(t){return void 0!==t&&(n.Array===t||i[o]===t)}},27904:(t,e,r)=>{var n=r(80812);t.exports=function(t,e,r,o){try{return o?e(n(r)[0],r[1]):e(r)}catch(e){var i=t.return;throw void 0!==i&&n(i.call(t)),e}}},30953:(t,e,r)=>{var n=r(2677);t.exports=function(t,e,r){for(var o in e)r&&t[o]?t[o]=e[o]:n(t,o,e[o]);return t}},643:(t,e,r)=>{var n=r(88535),o=r(25219),i=r(58852),a=r(95838);t.exports=function(t){n(n.S,t,{from:function(t){var e,r,n,u,s=arguments[1];return o(this),(e=void 0!==s)&&o(s),null==t?new this:(r=[],e?(n=0,u=i(s,arguments[2],2),a(t,!1,(function(t){r.push(u(t,n++))}))):a(t,!1,r.push,r),new this(r))}})}},44398:(t,e,r)=>{var n=r(88535);t.exports=function(t){n(n.S,t,{of:function(){for(var t=arguments.length,e=new Array(t);t--;)e[t]=arguments[t];return new this(e)}})}},6096:(t,e,r)=>{var n=r(24401);t.exports=function(t,e){if(!n(t)||t._t!==e)throw TypeError("Incompatible receiver, "+e+" required!");return t}},18355:(t,e,r)=>{var n=r(88535);n(n.S,"Array",{isArray:r(15461)})},77789:(t,e,r)=>{var n,o=r(66670),i=r(48219)(0),a=r(61331),u=r(75172),s=r(66854),f=r(2162),c=r(24401),l=r(6096),p=r(6096),d=!o.ActiveXObject&&"ActiveXObject"in o,v="WeakMap",h=u.getWeak,y=Object.isExtensible,b=f.ufstore,g=function(t){return function(){return t(this,arguments.length>0?arguments[0]:void 0)}},m={get:function(t){if(c(t)){var e=h(t);return!0===e?b(l(this,v)).get(t):e?e[this._i]:void 0}},set:function(t,e){return f.def(l(this,v),t,e)}},w=t.exports=r(99373)(v,g,m,f,!0,!0);p&&d&&(s((n=f.getConstructor(g,v)).prototype,m),u.NEED=!0,i(["delete","has","get","set"],(function(t){var e=w.prototype,r=e[t];a(e,t,(function(e,o){if(c(e)&&!y(e)){this._f||(this._f=new n);var i=this._f[t](e,o);return"set"==t?this:i}return r.call(this,e,o)}))})))},50836:(t,e,r)=>{r(643)("WeakMap")},78353:(t,e,r)=>{r(44398)("WeakMap")},68632:(t,e,r)=>{t.exports=r(37149)},73146:(t,e,r)=>{t.exports=r(85079)},82731:(t,e,r)=>{t.exports=r(3706)},5601:(t,e,r)=>{t.exports=r(80040)},692:(t,e,r)=>{var n=r(803).default,o=r(5601),i=r(66681),a=r(82731);function u(t){if("function"!=typeof o)return null;var e=new o,r=new o;return(u=function(t){return t?r:e})(t)}t.exports=function(t,e){if(!e&&t&&t.__esModule)return t;if(null===t||"object"!==n(t)&&"function"!=typeof t)return{default:t};var r=u(e);if(r&&r.has(t))return r.get(t);var o={},s=i&&a;for(var f in t)if("default"!==f&&Object.prototype.hasOwnProperty.call(t,f)){var c=s?a(t,f):null;c&&(c.get||c.set)?i(o,f,c):o[f]=t[f]}return o.default=t,r&&r.set(t,o),o},t.exports.__esModule=!0,t.exports.default=t.exports}}]);
//# sourceMappingURL=nteract_transforms_vsdom.bundle.js.map