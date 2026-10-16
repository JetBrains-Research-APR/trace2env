# AgentWorldBench os

An Ubuntu desktop driven by a computer-use agent on OSWorld-style tasks: mail folders and messages in Thunderbird,
spreadsheets, documents and slides in LibreOffice Calc, Writer and Impress, media settings and playlists in VLC,
browsing, bookmarks and settings in Chrome, image editing in GIMP, editor settings in VS Code, files on the Desktop
through the file manager and the terminal. The world model plays the desktop: the observation after every action is
the visible accessibility tree, and open windows, dialogs, menus, selections, typed text and files are the state.

Observation format (the text after `**Environment Observation:**`): one tab-separated table whose header line is
`tag	name	text	class	description	position (top-left x&y)	size (w&h)` followed by one line per visible
accessibility node, for example `push-button	New Message	New Message	button button-primary icon-button		(180, 118)	(134, 31)`.
Tags are ATK roles such as `push-button`, `static`, `table-cell`, `section`, `menu`, `menu-item`, `label`,
`toggle-button`, `paragraph`, `text`, `link`, `heading`, `list-item`, `entry`, `frame`, `dialog`; an empty text field is
written `""""""`; positions and sizes are screen pixels on a 1920x1080 desktop. The tree lists every application
window that is on screen, so switching or opening an app replaces most of the table, while a click inside a dialog
changes a few lines and leaves the rest in place.

Actions are free-form Python code blocks: `pyautogui.click(x, y)` (also `clicks`, `button='right'`), `pyautogui.write(text)`,
`pyautogui.press(key)` and `pyautogui.hotkey(*keys)`, `pyautogui.scroll(amount)`, `pyautogui.moveTo` / `dragTo`; agent
helpers `Agent.click(...)`, `Agent.type(text)`, `Agent.hotkey(...)`, `Agent.open_app(name)`, `key(...)`, `left_click(...)`;
application tools `BrowserTools.*`, `CalcTools.*`, `WriterTools.*`, `ImpressTools.*`, `VLCTools.*`; and arbitrary
`subprocess`, `os` or `json` calls. The executed code is the authoritative description of the action.

Behaviour: a click on a button, menu item or tab opens the target window, dialog or menu and may change the focused
application; typing fills the focused entry and the entry's text column shows it; hotkeys such as Ctrl+S, Ctrl+N or
Enter trigger the application's own command (save dialogs, new documents, confirmations); a menu closes when an item is
chosen or Escape is pressed; an action with no visible effect, such as a move, a hover or a click on an inert area,
leaves the tree unchanged; file operations in the terminal or through `os` change the file manager's listing only once
it is visible. Each transition is teacher-forced: the next observation is the real accessibility tree recorded after
the action.
